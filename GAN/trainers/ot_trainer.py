import os
import re
from abc import abstractmethod, ABC
from typing import Tuple

import torch
from torch import Tensor

import models
from configuration import Configuration
from trainers import AbstractBaseTrainer


def c_e_transform(y: torch.Tensor, C: torch.Tensor,
                  epsilon: float = 1) -> torch.Tensor:
    return softmin((C - y), epsilon=epsilon)[:None]


def c_e_tau_transform(y: torch.Tensor, C: torch.Tensor,
                      epsilon: float = 1, tau: float = 1):
    return aprox(softmin(C - y, epsilon), epsilon, tau)


def aprox(p: torch.Tensor, epsilon: float, tau: float) -> torch.Tensor:
    return (tau / (tau + epsilon)) * p


def softmin(X: torch.tensor, epsilon: float = 1) -> torch.Tensor:
    Y = -X / epsilon
    Ymax = torch.max(Y, 0)[0][:, None]
    return -epsilon * (
            Ymax + torch.log(torch.mean(torch.exp(Y - Ymax.t()), 0))[:,
                   None])


class AbstractBaseOTLossHelper(ABC):
    config: Configuration

    def __init__(self, config: Configuration):
        self.config = config

    @abstractmethod
    def dual_variable_transform(self, label_real: Tensor,
                                cost_matrix: Tensor) -> Tensor:
        pass

    @abstractmethod
    def objective_function(self, label_real: Tensor, label_fake: Tensor,
                           cost_matrix: Tensor) -> Tensor:
        pass


class BalancedOTLossHelper(AbstractBaseOTLossHelper):
    epsilon: float

    def __init__(self, config: Configuration):
        super().__init__(config)

        self.epsilon = self.config.loss.options['epsilon']

    def dual_variable_transform(self, label_real: Tensor,
                                cost_matrix: Tensor) -> Tensor:
        return c_e_transform(label_real, cost_matrix, epsilon=self.epsilon)

    def objective_function(self, label_real: Tensor, label_fake: Tensor,
                           cost_matrix: Tensor) -> Tensor:
        val0 = torch.mean(label_fake)
        val1 = torch.mean(label_real)

        tmp0 = (-cost_matrix + (label_fake + torch.t(label_real)))
        val_reg = self.epsilon * torch.mean(torch.exp(tmp0 / self.epsilon))

        val = val0 + val1 - val_reg
        return val


class UnbalancedOTLossHelper(AbstractBaseOTLossHelper):
    epsilon: float
    tau: float

    def __init__(self, config: Configuration):
        super().__init__(config)

        self.epsilon = self.config.loss.options['epsilon']
        self.tau = self.config.loss.options['tau']

    def dual_variable_transform(self, label_real: Tensor,
                                cost_matrix: Tensor) -> Tensor:
        return c_e_tau_transform(label_real, cost_matrix,
                                 epsilon=self.epsilon, tau=self.tau)

    def objective_function(self, label_real: Tensor, label_fake: Tensor,
                           cost_matrix: Tensor) -> Tensor:
        val_fake = self.tau * torch.mean(1 - torch.exp(-label_fake / self.tau))
        val_real = self.tau * torch.mean(1 - torch.exp(-label_real / self.tau))

        tmp0 = (-cost_matrix + (label_fake + label_real.T))
        val_reg = self.epsilon * torch.mean(torch.exp(tmp0 / self.epsilon))

        val = val_fake + val_real - val_reg
        return val


def lq_dist(x: torch.Tensor, y: torch.Tensor, p: int, q: int) -> torch.Tensor:
    return (torch.norm(x.unsqueeze(1) - y, dim=2, p=q) ** p) / p


class OTLossTrainer(AbstractBaseTrainer):
    ot_loss_helper: AbstractBaseOTLossHelper

    def __init__(self, config: Configuration):
        super().__init__(config)

        if config.loss.type == 'unbalanced':
            self.ot_loss_helper = UnbalancedOTLossHelper(config)
        elif config.loss.type == 'balanced':
            self.ot_loss_helper = BalancedOTLossHelper(config)
        else:
            raise Exception(f'Unknown loss type: {config.loss.type}')

    def _initialize_networks(self):
        self.generator_network = models.load_model(
            self.config.models.generator.type,
            latent_dim=self.config.train.latent_dimension,
            output_dim=self.dataset.data_dimension,
            **self.config.models.generator.options)

        self.discriminator_networks.append(models.load_model(
            self.config.models.discriminator.type,
            input_dim=self.dataset.data_dimension,
            **self.config.models.discriminator.options))

        if self.config.train.use_dual_critic_networks:
            self.discriminator_networks.append(models.load_model(
                self.config.models.discriminator.type,
                input_dim=self.dataset.data_dimension,
                **self.config.models.discriminator.options))

    def _load_checkpoints_dual(self, checkpoints_path: str) -> Tuple[int, int]:
        file_regex = re.compile(r'step_(\d+)_epoch_(\d+).pt')

        files = os.listdir(checkpoints_path)
        checkpoints = {}

        for file in files:
            match = file_regex.match(file)
            if not match:
                continue

            step = int(match.group(1))
            epoch = int(match.group(2))

            checkpoints[step] = (file, epoch)

        if not checkpoints:
            return 0, 0

        load_step = max(checkpoints.keys())

        load_epoch = checkpoints[load_step][1]

        checkpoint_path = os.path.join(checkpoints_path,
                                       checkpoints[load_step][0])
        checkpoint_dict = torch.load(checkpoint_path)

        self.generator_network.load_state_dict(checkpoint_dict['generator'])
        self.generator_network.train()

        self.discriminator_networks[0].load_state_dict(
            checkpoint_dict['discriminator0'])
        self.discriminator_networks[0].train()
        self.discriminator_networks[1].load_state_dict(
            checkpoint_dict['discriminator1'])
        self.discriminator_networks[1].train()

        return load_step, load_epoch

    def _load_checkpoints_single(self, checkpoints_path: str) \
            -> Tuple[int, int]:
        file_regex = re.compile(
            r'(discriminator|generator)_step_(\d+)_epoch_(\d+).pt')

        files = os.listdir(checkpoints_path)
        discriminator_checkpoints = {}
        generator_checkpoints = {}

        for file in files:
            match = file_regex.match(file)
            if not match:
                continue

            _type = match.group(1)
            step = int(match.group(2))
            epoch = int(match.group(3))

            if _type == 'discriminator':
                discriminator_checkpoints[step] = (file, epoch)
            else:
                generator_checkpoints[step] = (file, epoch)

        if not discriminator_checkpoints or not generator_checkpoints:
            return 0, 0

        max_step_discriminator = max(discriminator_checkpoints.keys())
        max_step_generator = max(discriminator_checkpoints.keys())

        load_step: int

        if max_step_discriminator != max_step_generator:
            print("WARNING: found generator and discriminator checkpoints "
                  "at different steps")
            load_step = min(max_step_discriminator, max_step_generator)
        else:
            load_step = max_step_discriminator

        load_epoch = discriminator_checkpoints[load_step][1]

        generator_checkpoint_path = os.path.join(
            checkpoints_path, generator_checkpoints[load_step][0])
        self.generator_network.load_state_dict(
            torch.load(generator_checkpoint_path))
        self.generator_network.train()

        discriminator_checkpoint_path = os.path.join(
            checkpoints_path, discriminator_checkpoints[load_step][0])
        self.discriminator_networks[0].load_state_dict(
            torch.load(discriminator_checkpoint_path))
        self.discriminator_networks[0].train()

        return load_step, load_epoch

    def _load_checkpoints(self, checkpoints_path: str) \
            -> Tuple[int, int]:
        print("Loading checkpoints")

        if self.config.train.use_dual_critic_networks:
            step, epoch = self._load_checkpoints_dual(checkpoints_path)
        else:
            step, epoch = self._load_checkpoints_single(checkpoints_path)

        if step == 0:
            print("No checkpoints available to load.")
            return 0, 0

        print(f"Loaded checkpoints at step {step} (epoch {epoch})")

        return step + 1, epoch

    def _save_checkpoints_single(self, checkpoints_path: str, epoch: int,
                                 step: int) -> None:
        torch.save(self.generator_network.state_dict(),
                   os.path.join(
                       checkpoints_path,
                       f'generator_step_{step}_epoch_{epoch}.pt'))
        torch.save(self.discriminator_networks[0].state_dict(),
                   os.path.join(
                       checkpoints_path,
                       f'discriminator_step_{step}_epoch_{epoch}.pt'))

    def _save_checkpoints_double(self, checkpoints_path: str, epoch: int,
                                 step: int) -> None:
        torch.save({
            'generator': self.generator_network.state_dict(),
            'discriminator0': self.discriminator_networks[0].state_dict(),
            'discriminator1': self.discriminator_networks[1].state_dict(),
        }, os.path.join(checkpoints_path, f'step_{step}_epoch_{epoch}.pt'))

    def _save_checkpoints(self, checkpoints_path: str, epoch: int,
                          step: int) -> None:
        if self.config.train.use_dual_critic_networks:
            self._save_checkpoints_double(checkpoints_path, epoch, step)
        else:
            self._save_checkpoints_single(checkpoints_path, epoch, step)

    def _get_single_loss(self, data_fake: Tensor,
                         cost_matrix_cross: Tensor) -> Tensor:
        label_fake = self.discriminator_networks[0](data_fake)
        label_real_transformed_fake = self.ot_loss_helper \
            .dual_variable_transform(label_fake, cost_matrix_cross)

        if self.config.train.use_double_dual_transform:
            label_fake_double_transformed = self.ot_loss_helper \
                .dual_variable_transform(label_real_transformed_fake,
                                         cost_matrix_cross.T)

            label_fake = label_fake_double_transformed

        loss_val = self.ot_loss_helper.objective_function(
            label_real_transformed_fake, label_fake, cost_matrix_cross)

        return loss_val

    def _get_discriminator_loss(self,
                                discriminator_index: int,
                                batch_size_real: int,
                                batch_size_fake: int,
                                data_real: Tensor) -> Tensor:

        data_real_flat = data_real.reshape(batch_size_real, -1)

        img_shape = data_real.shape[1:]

        with torch.no_grad():
            z = torch.randn((batch_size_fake,
                             self.config.train.latent_dimension),
                            device=data_real.device)
            data_fake = self.generator_network(z).reshape(-1, *img_shape)

            data_fake_flat = data_fake.reshape(batch_size_fake, -1)
            cost_matrix_cross = lq_dist(data_fake_flat, data_real_flat, 2, 2)

        if self.config.train.use_dual_critic_networks:
            label_fake = self.discriminator_networks[0](data_fake)
            label_real_transformed_fake = self.ot_loss_helper \
                .dual_variable_transform(label_fake, cost_matrix_cross)

            label_real = self.discriminator_networks[1](data_real)
            label_fake_transformed_real = self.ot_loss_helper \
                .dual_variable_transform(label_real, cost_matrix_cross.T)

            penalty_factor = 0.5 * 10 ** -1

            if discriminator_index == 0:
                transform_penalty = penalty_factor * (
                        label_fake - label_fake_transformed_real).pow(2).sum()

                loss_val = self.ot_loss_helper.objective_function(
                    label_real_transformed_fake,
                    label_fake, cost_matrix_cross)

                print("Raw loss:", loss_val.item())
                print("Penalty:", transform_penalty.item())

                loss_val -= transform_penalty
            else:
                transform_penalty = penalty_factor * (
                        label_real - label_real_transformed_fake).pow(2).sum()

                loss_val = self.ot_loss_helper.objective_function(
                    label_fake_transformed_real,
                    label_real, cost_matrix_cross.T)

                print("Raw loss:", loss_val.item())
                print("Penalty:", transform_penalty.item())

                loss_val -= transform_penalty
        else:
            loss_val = self._get_single_loss(data_fake, cost_matrix_cross)

        return -loss_val

    def _get_generator_loss(self, batch_size_real: int, batch_size_fake: int,
                            data_real: Tensor) -> Tensor:
        data_real_flat = data_real.reshape(batch_size_real, -1)

        img_shape = data_real.shape[1:]

        z = torch.randn((batch_size_fake,
                         self.config.train.latent_dimension),
                        device=data_real.device)
        data_fake = self.generator_network(z).reshape(-1, *img_shape)

        data_fake_flat = data_fake.reshape(batch_size_fake, -1)
        cost_matrix_cross = lq_dist(data_fake_flat, data_real_flat, 2, 2)

        if self.config.train.use_dual_critic_networks:
            label_fake = self.discriminator_networks[0](data_fake)
            label_real = self.discriminator_networks[1](data_real)

            loss_val = self.ot_loss_helper.objective_function(
                label_real, label_fake, cost_matrix_cross)
        else:
            loss_val = self._get_single_loss(data_fake, cost_matrix_cross)

        return loss_val

