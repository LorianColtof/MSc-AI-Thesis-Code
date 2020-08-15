import copy
import hashlib
import json
import os
from functools import wraps
from typing import Callable, Any, Optional, NamedTuple, Union

import mlflow

from configuration import Configuration

AnyCallable = Callable[..., Any]


def enable_mlflow_tracking_class(config_property: str) \
        -> Callable[[AnyCallable], AnyCallable]:

    def real_decorator(f: AnyCallable) -> AnyCallable:
        @wraps(f)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            config: Configuration = getattr(self, config_property)

            @enable_mlflow_tracking(config)
            def wrapper_with_config(self: Any, *args: Any,
                                    **kwargs: Any) -> Any:
                f(self, *args, **kwargs)

            return wrapper_with_config(self, *args, **kwargs)

        return wrapper

    return real_decorator


def enable_mlflow_tracking(config: Configuration) \
        -> Callable[[AnyCallable], AnyCallable]:

    def real_decorator(f: AnyCallable) -> AnyCallable:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if config.train.mlflow.enabled:
                if config.train.mlflow.tracking_server_url:
                    mlflow.set_tracking_uri(
                        config.train.mlflow.tracking_server_url)

                mlflow.set_experiment(config.train.mlflow.experiment_name)

                config_dict = copy.deepcopy(
                    config.runtime_options['config_dict'])

                # Remove some config keys which we should be able to change
                # without giving a different the checksum

                for key in ['maximum_epochs', 'maximum_steps',
                            'use_checkpoints', 'output_directory',
                            'save_interval', 'mlflow']:
                    if key in config_dict['train']:
                        del config_dict['train'][key]

                del config_dict['dataset']['directory']

                config_checksum = hashlib.sha256(
                    json.dumps(config_dict,
                               sort_keys=True).encode('utf-8')).hexdigest()

                print(f"Config checksum: {config_checksum}")
                matching_run_id: Optional[str] = None

                if config.train.use_checkpoints:
                    print("Checking for runs to continue from")

                    current_experiment = mlflow.get_experiment_by_name(
                        config.train.mlflow.experiment_name)
                    runs = mlflow.search_runs(
                        experiment_ids=[current_experiment.experiment_id])

                    for _, run in runs.iterrows():
                        run_config_checksum = run['tags.config_checksum']
                        run_id = run['run_id']
                        if config_checksum == run_config_checksum:
                            print(f'Run {run_id} matches.')
                            matching_run_id = run_id
                            break
                        elif not run_config_checksum:
                            print(f'Run {run_id} misses config checksum.')
                        else:
                            print(f'Run {run_id} does not match.')
                else:
                    print("Not using checkpoints, so creating a new run.")

                with mlflow.start_run(run_id=matching_run_id,
                                      run_name=config.train.mlflow.run_name):
                    if not matching_run_id:
                        mlflow.set_tag('config_checksum', config_checksum)

                        _log_configuration(config)

                    slurm_jobid = os.environ.get('SLURM_JOBID')
                    if slurm_jobid:
                        mlflow.set_tag('slurm_jobid', slurm_jobid)

                    f(*args, **kwargs)
            else:
                f(*args, **kwargs)

        return wrapper

    return real_decorator


def _log_configuration(config: Configuration) -> None:
    _log_configuration_params(config)
    if config.runtime_options['config_filename']:
        mlflow.log_artifact(config.runtime_options['config_filename'])


def _log_configuration_params(config_item: Union[NamedTuple, dict],
                              prefix: Optional[str] = '') -> None:
    if prefix == 'runtime_options.config_dict.':
        return

    if isinstance(config_item, tuple):
        config_item = config_item._asdict()

    for key, value in config_item.items():
        if isinstance(value, tuple) or isinstance(value, dict):
            _log_configuration_params(value, f'{prefix}{key}.')
        else:
            mlflow.log_param(f'{prefix}{key}', value)
