from typing import IO, Dict, Any, NamedTuple, TypedDict, Optional

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

config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Configuration file",
    "type": "object",
    "required": ["dataset", "models", "optimizers", "loss", "train"],
    "properties": {
        "dataset": {
            "type": "object",
            "required": ["type", "directory"],
            "properties": {
                "type": {"type": "string"},
                "directory": {"type": "string"},
            }
        },
        "models": {
            "type": "object",
            "required": ["generator", "discriminator"],
            "properties": {
                "generator": type_with_options_schema,
                "discriminator": type_with_options_schema
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
            "required": ["save_interval", "maximum_epochs", "maximum_steps"],
            "default": {
                "critic_steps": default_critic_steps
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
                "mlflow": {
                    "type": "object",
                    "required": ["enabled", "experiment_name"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "experiment_name": {"type": "string"},
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
    directory: str


class TypeWithOptions(NamedTuple):
    type: str
    options: Dict[str, Any] = {}


class GeneratorDiscriminator(NamedTuple):
    generator: TypeWithOptions
    discriminator: TypeWithOptions


class MLflow(NamedTuple):
    enabled: bool = False
    experiment_name: Optional[str] = None
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


class RuntimeOptions(TypedDict):
    device: torch.device
    config_filename: Optional[str]
    config_dict: dict


class Configuration(NamedTuple):
    dataset: Dataset
    models: GeneratorDiscriminator
    optimizers: GeneratorDiscriminator
    loss: TypeWithOptions
    train: Train
    runtime_options: RuntimeOptions = RuntimeOptions(
        device=torch.device('cpu'), config_filename=None)


def _create_generator_discriminator_info(
        data: Dict[str, Any]) -> GeneratorDiscriminator:
    generator = TypeWithOptions(**data['generator'])
    discriminator = TypeWithOptions(**data['discriminator'])

    return GeneratorDiscriminator(generator, discriminator)


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
