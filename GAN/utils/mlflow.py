from typing import Callable, Any, Optional, NamedTuple, Union

import mlflow
from configuration import Configuration

from functools import wraps

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

                with mlflow.start_run():
                    _log_configuration(config)
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
    if isinstance(config_item, tuple):
        config_item = config_item._asdict()

    for key, value in config_item.items():
        if isinstance(value, tuple) or isinstance(value, dict):
            _log_configuration_params(value, f'{prefix}{key}.')
        else:
            mlflow.log_param(f'{prefix}{key}', value)
