import sys
from typing import IO, Dict, Any, NamedTuple, Optional

if sys.version_info >= (3, 8):
    from typing import TypedDict

import jsonschema
import torch
import yaml

type_with_options_schema = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {"type": "string"},
        "options": {
            "type": "object"
        }
    }
}

default_critic_steps = 1
default_num_samples = 10

config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Configuration file",
    "type": "object",
    "required": ["dataset", "models", "optimizers", "loss", "train"],
    "properties": {
        "dataset": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"type": "string"},
                "directory": {"type": "string"},
                "source_class": {"type": "string"}
            }
        },
        "models": {
            "type": "object",
            "required": ["generator", "discriminator"],
            "properties": {
                "generator": type_with_options_schema,
                "discriminator": type_with_options_schema,
                "source_encoder": type_with_options_schema
            }
        },
        "optimizers": {
            "type": "object",
            "required": ["generator", "discriminator"],
            "properties": {
                "generator": type_with_options_schema,
                "discriminator": type_with_options_schema
            }
        },
        "loss": type_with_options_schema,
        "train": {
            "type": "object",
            "required": ["type", "save_interval",
                         "maximum_epochs", "maximum_steps"],
            "default": {
                "critic_steps": default_critic_steps,
                "num_samples": default_num_samples
            },
            "properties": {
                "type": {"type": "string"},
                "output_directory": {"type": "string"},
                "save_interval": {"type": "integer"},
                "maximum_epochs": {"type": "integer"},
                "maximum_steps": {"type": "integer"},
                "critic_steps": {"type": "integer"},
                "batch_size_fake": {"type": "integer"},
                "use_dual_critic_networks": {"type": "boolean"},
                "use_checkpoints": {"type": "boolean"},
                "use_double_dual_transform": {"type": "boolean"},
                "num_samples": {"type": "integer"},
                "mlflow": {
                    "type": "object",
                    "required": ["enabled", "experiment_name"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "experiment_name": {"type": "string"},
                        "run_name": {"type": "string"},
                        "tracking_server_url": {"type": "string"}
                    }
                }
            }

        },
    }
}

config_validator = jsonschema.Draft7Validator(config_schema)


class Dataset(NamedTuple):
    type: str
    directory: Optional[str] = None
    source_class: Optional[str] = None


class TypeWithOptions(NamedTuple):
    type: str
    options: Dict[str, Any] = {}


class Models(NamedTuple):
    generator: TypeWithOptions
    discriminator: TypeWithOptions
    source_encoder: Optional[TypeWithOptions] = None


class MLflow(NamedTuple):
    enabled: bool = False
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tracking_server_url: Optional[str] = None
    artifact_uri: Optional[str] = None


class Train(NamedTuple):
    type: str
    batch_size: int
    latent_dimension: int
    save_interval: int
    maximum_epochs: int
    maximum_steps: int
    mlflow: MLflow
    output_directory: Optional[str] = None
    critic_steps: int = default_critic_steps
    batch_size_fake: Optional[int] = None
    use_dual_critic_networks: bool = False
    use_checkpoints: bool = True
    use_double_dual_transform: bool = True
    num_samples: int = default_num_samples


if sys.version_info >= (3, 8):
    class RuntimeOptions(TypedDict):
        device: torch.device
        config_filename: Optional[str]
        config_dict: dict
        dataloader_num_workers: int
else:
    RuntimeOptions = dict


class Configuration(NamedTuple):
    dataset: Dataset
    models: Models
    optimizers: Models
    loss: TypeWithOptions
    train: Train
    runtime_options: RuntimeOptions = RuntimeOptions(
        device=torch.device('cpu'), config_filename=None)


def _create_generator_discriminator_info(
        data: Dict[str, Any]) -> Models:
    generator = TypeWithOptions(**data['generator'])
    discriminator = TypeWithOptions(**data['discriminator'])
    source_encoder = TypeWithOptions(**data['source_encoder']) \
        if 'source_encoder' in data else None

    return Models(generator, discriminator, source_encoder)


def load_configuration(config_file: IO) -> Configuration:
    config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    config_validator.validate(config)

    dataset = Dataset(**config['dataset'])
    models = _create_generator_discriminator_info(config['models'])
    optimizers = _create_generator_discriminator_info(config['optimizers'])
    loss = TypeWithOptions(**config['loss'])

    if 'mlflow' in config['train']:
        config['train']['mlflow'] = MLflow(**config['train']['mlflow'])
    else:
        config['train']['mlflow'] = MLflow(enabled=False)

    train = Train(**config['train'])

    config_result = Configuration(dataset, models, optimizers, loss, train)
    config_result.runtime_options['config_dict'] = config

    return config_result
