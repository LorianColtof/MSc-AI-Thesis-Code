import argparse

import torch

import configuration
from trainers import AbstractBaseTrainer
from trainers.mswd_trainer import MaxSlicedWassersteinLossTrainer
from trainers.ot_trainer import OTLossTrainer
from trainers.wgan_gp_trainer import WassersteinGPLossTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=argparse.FileType('r'),
                        required=True)

    args = parser.parse_args()

    config = configuration.load_configuration(args.config_file)

    if torch.cuda.is_available():
        print("Using CUDA")
        config.runtime_options['device'] = torch.device('cuda')
    else:
        print("CUDA is not available, falling back to CPU")
        config.runtime_options['device'] = torch.device('cpu')

    trainer: AbstractBaseTrainer

    if config.train.type == 'ot_gan':
        trainer = OTLossTrainer(config)
    elif config.train.type == 'mswd_gan':
        trainer = MaxSlicedWassersteinLossTrainer(config)
    elif config.train.type == 'wgan_gp':
        trainer = WassersteinGPLossTrainer(config)
    else:
        raise Exception(f'Invalid training type: {config.train.type}')

    trainer.train()


if __name__ == "__main__":
    main()
