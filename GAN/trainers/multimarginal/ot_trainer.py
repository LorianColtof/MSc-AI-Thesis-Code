from typing import List, Dict, Iterator, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import models
from configuration import Configuration
from . import AbstractMultimarginalBaseTrainer


class MultimarginalOTLossTrainer(AbstractMultimarginalBaseTrainer):
    def __init__(self, config: Configuration):
        super().__init__(config)

        self.weight_cls = self.config.loss.options.get('weight_cls', 1)
        self.weight_mut_info = self.config.loss.options.get(
            'weight_mut_info', 1)
        self.weight_idt = self.config.loss.options.get('weight_idt', 0)
        self.cls_loss_type = self.config.loss.type

        self.epsilon = self.config.loss.options['epsilon']
        self.enable_adaptive_reg_param = self.config.loss.options.get(
            'enable_adaptive_reg_param', False)
        self.enable_log_reg_term = self.config.loss.options.get(
            'enable_log_reg_term', False)

        if self.enable_adaptive_reg_param and self.enable_log_reg_term:
            raise Exception('Cannot have both enable_adaptive_reg_param '
                            'and enable_log_reg_term set to true.')

        # Limit for each term in the regularization term (exp(L))
        # before applying regularization parameter adaption
        self.adaptive_reg_param_sum_limit = self.config.loss.options.get(
            'adaptive_reg_param_sum_limit', 50)

        if self.cls_loss_type not in {'LS', 'BCE'}:
            raise Exception('cls_loss_type must be one of: LS, BCE')

        dist_params = list(
            self.config.loss.options.get('dist_params', [2, 2]))
        if len(dist_params) != 2:
            raise Exception('dist_params should be a list of 2 floats')

        self.dist_q = dist_params[0]
        self.dist_p = dist_params[1]

        self.idt_loss_criterion = torch.nn.L1Loss()

        self.use_one_generator_optimizer = True

    def _initialize_networks(self):
        if not self.config.models.source_encoder:
            raise Exception("models.source_encoder is required")

        self.encoder_network = models.load_model(
            self.config.models.source_encoder.type,
            latent_dim=self.config.train.latent_dimension,
            output_dim=self.dataset.data_dimension,
            **self.config.models.source_encoder.options)

        self.generator_networks = [
            models.load_model(
                self.config.models.generator.type,
                latent_dim=self.config.train.latent_dimension,
                output_dim=self.dataset.data_dimension,
                **self.config.models.generator.options)
            for _ in range(len(self.dataset.target_classes))
        ]

        self.discriminator_networks.append(models.load_model(
            self.config.models.discriminator.type,
            input_dim=self.dataset.data_dimension,
            num_classes=len(self.dataset.target_classes),
            **self.config.models.discriminator.options))

    def _classification_loss(self, logit: Tensor, target: Tensor) -> Tensor:
        if self.cls_loss_type == 'BCE':
            return F.binary_cross_entropy_with_logits(logit, target)
        else:
            return F.mse_loss(logit, target)

    def _data_distance(self, data_list: List[Tensor]) \
            -> Tensor:
        num_dimensions = len(data_list)
        data_unsqueezed_list = []
        device = data_list[0].device
        batch_size = data_list[0].size(0)

        for dim, data in enumerate(data_list):
            shape = ([1] * dim + [batch_size]
                     + [1] * (num_dimensions - dim - 1) + [-1])
            data_unsqueezed_list.append(data.reshape(*shape))

        norm_sum = torch.zeros([batch_size] * num_dimensions, device=device)
        for i, d1 in enumerate(data_unsqueezed_list):
            for d2 in data_unsqueezed_list[i + 1:]:
                norm = torch.norm(d1 - d2,
                                  dim=-1, p=self.dist_q) ** self.dist_p
                norm_sum += norm

        return norm_sum / (num_dimensions * self.dist_p)

    def _get_adversarial_loss_reg(self, data_source: Tensor,
                                  data_fake_list: List[Tensor],
                                  discriminators_fake: List[Tensor],
                                  batch_size: int) -> Tuple[Tensor, Tensor]:
        num_target_classes = len(data_fake_list)

        disc_real_out, _ = self.discriminator_networks[0](data_source)

        # B x 1
        disc_real_out = disc_real_out \
            .reshape(batch_size, -1).mean(dim=1)

        # B x ... x B   (num_target_classes + 1 times)
        cost_tensor = self._data_distance([data_source] + data_fake_list)

        reg_sum = -cost_tensor + disc_real_out.reshape(
            [-1] + [1] * num_target_classes)
        for dim, disc_fake_out in enumerate(discriminators_fake):
            reg_sum_shape = ([1] * (dim + 1) + [-1] +
                             [1] * (num_target_classes - dim - 1))
            reg_sum += disc_fake_out.reshape(*reg_sum_shape)

        epsilon: float
        adv_loss_reg: Tensor

        if self.enable_adaptive_reg_param:
            epsilon_low = self.epsilon
            sum_limit_normalized = \
                self.adaptive_reg_param_sum_limit * epsilon_low

            reg_sum_max = reg_sum.max()

            if reg_sum_max > sum_limit_normalized:
                epsilon = reg_sum_max / self.adaptive_reg_param_sum_limit
            else:
                epsilon = epsilon_low
        else:
            epsilon = self.epsilon

        if self.enable_log_reg_term:
            reg_sum_epsilon = reg_sum / epsilon
            item_max = reg_sum_epsilon.max()
            adv_loss_reg = epsilon * (
                    item_max + (reg_sum_epsilon - item_max).exp().mean().log())
        else:
            adv_loss_reg = epsilon * torch.exp(reg_sum / epsilon).mean()

        return adv_loss_reg, disc_real_out

    def _get_multiclass_discriminator_loss(
            self, discriminator_index: int,
            batch_size: int, data_source: Tensor,
            data_targets: List[Tensor]) -> Tensor:
        device = data_source.device
        num_target_classes = len(data_targets)

        with torch.no_grad():
            embedding = self.encoder_network(data_source)

        total_classification_loss = torch.zeros(1, device=device)
        total_adversarial_loss_targets = torch.zeros(1, device=device)

        label_pos = torch.tensor([1.0] * batch_size, device=device)
        label_neg = torch.tensor([0.0] * batch_size, device=device)

        discriminators_fake = []
        data_fake_list = []

        for i, generator in enumerate(self.generator_networks):
            with torch.no_grad():
                data_fake = generator(embedding)

            data_fake_list.append(data_fake)

            disc_fake_out, disc_fake_out_class = \
                self.discriminator_networks[0](data_fake)
            _, disc_real_out_class = self.discriminator_networks[0](
                data_targets[i])

            # B x 1
            disc_fake_out = disc_fake_out\
                .reshape(batch_size, -1).mean(dim=1)

            total_classification_loss += \
                (self._classification_loss(
                    disc_real_out_class[:, i], label_pos) +
                 self._classification_loss(
                     disc_fake_out_class[:, i], label_neg))

            total_adversarial_loss_targets += disc_fake_out.mean()

            discriminators_fake.append(disc_fake_out)

        adv_loss_reg, disc_real_out = self._get_adversarial_loss_reg(
            data_source, data_fake_list, discriminators_fake, batch_size)

        total_adversarial_loss = (
                disc_real_out.mean() +
                total_adversarial_loss_targets / num_target_classes
                - adv_loss_reg)

        total_loss = (-total_adversarial_loss
                      + self.weight_cls * total_classification_loss)

        return total_loss

    def _train_multiclass_generators(
            self, batch_size: int, data_source: Tensor,
            target_data_iterators: Dict[str, Iterator[Tensor]]):

        device = data_source.device
        num_target_classes = len(target_data_iterators.keys())

        embedding = self.encoder_network(data_source)

        total_mut_info_loss = torch.zeros(1, device=device)
        total_adversarial_loss_targets = torch.zeros(1, device=device)

        total_idt_loss = torch.zeros(1, device=device)
        label_pos = torch.tensor([1.0] * batch_size, device=device)

        discriminators_fake = []
        data_fake_list = []

        for i, generator in enumerate(self.generator_networks):
            data_fake = generator(embedding)

            data_fake_list.append(data_fake)

            disc_fake_out, disc_fake_out_class = \
                self.discriminator_networks[0](data_fake)

            # B x 1
            disc_fake_out = disc_fake_out \
                .reshape(batch_size, -1).mean(dim=1)

            if self.weight_idt > 0:
                _class = self.dataset.target_classes[i]
                data_target = next(target_data_iterators[_class])[0].to(device)
                idt_fake = generator(self.encoder_network(data_target))
                idt_loss = self.idt_loss_criterion(idt_fake, data_target)

                total_idt_loss += idt_loss

            total_mut_info_loss += F.binary_cross_entropy_with_logits(
                disc_fake_out_class[:, i], label_pos)
            total_adversarial_loss_targets += disc_fake_out.mean()

            discriminators_fake.append(disc_fake_out)

        adv_loss_reg, disc_real_out = self._get_adversarial_loss_reg(
            data_source, data_fake_list, discriminators_fake, batch_size)

        # Skip disc_real_out.mean() since it's constant
        # w.r.t. generator parameters
        total_adversarial_loss = (
                total_adversarial_loss_targets / num_target_classes
                - adv_loss_reg)

        total_loss = (total_adversarial_loss
                      + self.weight_mut_info * total_mut_info_loss
                      + self.weight_idt * total_idt_loss)

        self._log_loss('Generator loss', 'generator_loss', total_loss)

        self._optimize_generator(total_loss, self.generator_optimizers[0])


