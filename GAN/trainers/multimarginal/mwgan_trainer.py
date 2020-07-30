from typing import Dict, Iterator, List

import torch
from torch import Tensor
import torch.nn.functional as F

import models
from configuration import Configuration
from . import AbstractMultimarginalBaseTrainer


class MultimarginalWassersteinGPLossTrainer(AbstractMultimarginalBaseTrainer):
    def __init__(self, config: Configuration):
        super().__init__(config)

        Lf = self.config.loss.options['const_inter_domain']
        if not Lf:
            Lf = 1

        lambda_cls = self.config.loss.options['const_cls']
        if not lambda_cls:
            lambda_cls = 1

        lambda_reg = self.config.loss.options['const_reg']
        if not lambda_reg:
            lambda_reg = 100

        self.Lf = Lf
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg


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
        return F.binary_cross_entropy_with_logits(logit, target)

    def _get_multiclass_discriminator_loss(
            self, discriminator_index: int,
            batch_size: int, data_source: Tensor,
            data_targets: List[Tensor]) -> Tensor:

        device = data_source.device
        num_target_classes = len(data_targets)

        with torch.no_grad():
            embedding = self.encoder_network(data_source)

        total_classification_loss = torch.zeros(1, device=device)
        total_adversarial_loss = torch.zeros(1, device=device)

        label_pos = torch.tensor([1.0] * batch_size, device=device)
        label_neg = torch.tensor([0.0] * batch_size, device=device)

        data_fake_list = []

        for i, generator in enumerate(self.generator_networks):
            with torch.no_grad():
                data_fake = generator(embedding)

            data_fake_list.append(data_fake)

            disc_fake_out, disc_fake_out_class = self.discriminator_networks[0](
                data_fake)
            _, disc_real_out_class = self.discriminator_networks[0](
                data_targets[i])

            total_classification_loss += \
                (self._classification_loss(
                    disc_fake_out_class[:, i], label_pos) +
                 self._classification_loss(
                     disc_real_out_class[:, i], label_neg))
            total_adversarial_loss += disc_fake_out.mean()

        disc_real_out, disc_real_out_class = self.discriminator_networks[0](
            data_source)

        total_adversarial_loss = (disc_real_out.mean() -
                                  total_adversarial_loss / num_target_classes)

        data_fake_cat = torch.cat(data_fake_list).data
        data_source_cat = torch.cat([data_source] * num_target_classes).data

        alpha = torch.rand(data_source_cat.size(0), 1, 1, 1, device=device)

        data_interp = (alpha * data_source_cat + (1 - alpha) * data_fake_cat)\
            .requires_grad_(True)

        out_interp, _ = self.discriminator_networks[0](data_interp)

        gradients = torch.autograd.grad(
            outputs=out_interp, inputs=data_interp,
            grad_outputs=torch.ones(out_interp.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients_norm = gradients.norm(2, dim=1)

        zeros = torch.zeros_like(gradients_norm, device=device)
        penalty = torch.max(gradients_norm - self.Lf, zeros).mean() ** 2

        total_loss = (-total_adversarial_loss
                      + self.lambda_cls * total_classification_loss
                      + self.lambda_reg * penalty)

        return total_loss

    def _train_multiclass_generators(
            self, batch_size: int, data_source: Tensor,
            target_data_iterators: Dict[str, Iterator[Tensor]]) -> Tensor:

        device = data_source.device
        num_target_classes = len(target_data_iterators)

        embedding = self.encoder_network(data_source)

        total_mut_info_loss = torch.zeros(1, device=device)
        total_adversarial_loss = torch.zeros(1, device=device)

        # TODO: add identity loss
        # total_idt_loss = torch.zeros(1, device=device)

        label_pos = torch.tensor([1.0] * batch_size, device=device)

        for i, generator in enumerate(self.generator_networks):
            data_fake = generator(embedding)

            disc_fake_out, disc_fake_out_class = self.discriminator_networks[0](
                data_fake)

            total_mut_info_loss += F.binary_cross_entropy_with_logits(
                disc_fake_out_class[:, i], label_pos)
            total_adversarial_loss -= disc_fake_out.mean()

        # TODO: create hyperparameter
        lambda_mut_info = 1

        total_loss = (total_adversarial_loss / num_target_classes
                      + lambda_mut_info * total_mut_info_loss)

        self._log_loss('Generator loss', 'generator_loss', total_loss)

        self._optimize_generator(total_loss, self.generator_optimizers[0])

