import argparse

import torch
import sys

import configuration
from trainers import AbstractBaseTrainer
from trainers.mswd_trainer import MaxSlicedWassersteinLossTrainer, \
    SampledSlicedWassersteinLossTrainer
from trainers.ot_trainer import OTLossTrainer
from trainers.wgan_gp_trainer import WassersteinGPLossTrainer
from trainers.multimarginal.mwgan_trainer \
    import MultimarginalWassersteinGPLossTrainer
from trainers.multimarginal.ot_trainer import MultimarginalOTLossTrainer


def debugger_active() -> bool:
    return sys.gettrace() is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=argparse.FileType('r'),
                        required=True)
    parser.add_argument('--no-cuda', action='store_true')

    args = parser.parse_args()

    config = configuration.load_configuration(args.config_file)

    config.runtime_options['device'] = torch.device('cpu')

    if torch.cuda.is_available():
        if args.no_cuda:
            print("CUDA is disabled. Using CPU.")
        else:
            print("Using CUDA")
            config.runtime_options['device'] = torch.device('cuda')
    else:
        print("CUDA is not available, falling back to CPU")

    if debugger_active():
        print("Debugger is active, using num_workers=0 for DataLoaders")
        config.runtime_options['dataloader_num_workers'] = 0
    else:
        config.runtime_options['dataloader_num_workers'] = 4

    config.runtime_options['config_filename'] = args.config_file.name

    torch.random.manual_seed(42)

    trainer: AbstractBaseTrainer

    if config.train.type == 'ot_gan':
        trainer = OTLossTrainer(config)
    elif config.train.type == 'mswd_gan':
        trainer = MaxSlicedWassersteinLossTrainer(config)
    elif config.train.type == 'sampled_swd_gan':
        trainer = SampledSlicedWassersteinLossTrainer(config)
    elif config.train.type == 'wgan_gp':
        trainer = WassersteinGPLossTrainer(config)
    elif config.train.type == 'mwgan_gp':
        trainer = MultimarginalWassersteinGPLossTrainer(config)
    elif config.train.type == 'm_ot_gan':
        trainer = MultimarginalOTLossTrainer(config)
    else:
        raise Exception(f'Invalid training type: {config.train.type}')

    trainer.train()


if __name__ == "__main__":
    main()
