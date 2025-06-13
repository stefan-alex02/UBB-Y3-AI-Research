import importlib
import itertools
from typing import Dict, List, Any, Union, Type, Optional

import torch
from skorch.callbacks import LRScheduler

from model_src.server.ml.logger_utils import logger

OPTIMIZER_MAP: Dict[str, Type[torch.optim.Optimizer]] = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

DEFAULT_LR_SCHEDULER_NAME = 'default_lr_scheduler'


def _resolve_optimizer(opt_val_or_list: Any) -> Any:
    """Converts optimizer string(s) to type(s)."""
    if isinstance(opt_val_or_list, list):
        return [OPTIMIZER_MAP.get(opt.lower(), opt) if isinstance(opt, str) else opt for opt in opt_val_or_list]
    elif isinstance(opt_val_or_list, str):
        return OPTIMIZER_MAP.get(opt_val_or_list.lower(), opt_val_or_list)
    return opt_val_or_list

def _resolve_scheduler_policy_class(policy_name_or_class: Union[str, Type]) -> Type:
    """
    Resolves a scheduler policy string (e.g., "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts")
    or a class to the actual class type.
    """
    if isinstance(policy_name_or_class, str):
        if '.' in policy_name_or_class: # Fully qualified path
            try:
                module_path, class_name = policy_name_or_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"Could not import scheduler class '{policy_name_or_class}': {e}"
                )
        else:
            try:
                return getattr(torch.optim.lr_scheduler, policy_name_or_class)
            except AttributeError:
                raise ValueError(
                    f"Scheduler policy string '{policy_name_or_class}' not found in torch.optim.lr_scheduler "
                    f"and not a fully qualified path."
                )
    elif isinstance(policy_name_or_class, type) and issubclass(policy_name_or_class, torch.optim.lr_scheduler._LRScheduler):
        return policy_name_or_class
    else:
        raise TypeError(
            f"Scheduler policy must be a string (name or full path) or a _LRScheduler subclass, got {type(policy_name_or_class)}"
        )


def parse_fixed_hyperparameters(
        fixed_params: Dict[str, Any],
        default_max_epochs_for_cosine: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parses a dictionary of fixed hyperparameters:
    - Resolves optimizer string to its type.
    - Constructs the LRScheduler callback object from its policy string and params.
    - Prepares other callback parameters if specified.

    Args:
        fixed_params: Dictionary of hyperparameters.
                      'optimizer' can be a string.
                      'callbacks__<name>__policy' and related scheduler params are expected.
        default_max_epochs_for_cosine: Used for T_max in CosineAnnealingLR if not specified.

    Returns:
        A dictionary of processed hyperparameters ready for SkorchModelAdapter.
    """
    processed_params = fixed_params.copy()

    if 'optimizer' in processed_params:
        opt_val = processed_params['optimizer']
        if isinstance(opt_val, str):
            opt_type = OPTIMIZER_MAP.get(opt_val.lower())
            if opt_type is None:
                raise ValueError(f"Unsupported optimizer string in fixed_params: '{opt_val}'")
            processed_params['optimizer'] = opt_type
        elif not (isinstance(opt_val, type) and issubclass(opt_val, torch.optim.Optimizer)):
            raise TypeError(f"Optimizer in fixed_params must be a string or Optimizer type, got {type(opt_val)}")

    scheduler_key_prefix = f'callbacks__{DEFAULT_LR_SCHEDULER_NAME}__'
    policy_key = f'{scheduler_key_prefix}policy'

    if policy_key in processed_params:
        policy_name_or_class_from_params = processed_params.pop(policy_key)

        try:
            actual_scheduler_class = _resolve_scheduler_policy_class(policy_name_or_class_from_params)
        except ValueError as e:
            logger.error(f"Error resolving scheduler policy: {e}")
            raise

        scheduler_constructor_kwargs: Dict[str, Any] = {}
        direct_lr_scheduler_args: Dict[str, Any] = {}

        param_keys_for_this_scheduler = [
            k for k in list(processed_params.keys()) if k.startswith(scheduler_key_prefix)
        ]

        for key in param_keys_for_this_scheduler:
            param_name = key.split('__')[-1]
            if param_name in ['monitor', 'event_name', 'step_every']:
                direct_lr_scheduler_args[param_name] = processed_params.pop(key)
            else:
                scheduler_constructor_kwargs[param_name] = processed_params.pop(key)

        if actual_scheduler_class == torch.optim.lr_scheduler.CosineAnnealingLR and 'T_max' not in scheduler_constructor_kwargs:
            if default_max_epochs_for_cosine:
                scheduler_constructor_kwargs['T_max'] = default_max_epochs_for_cosine
            else:
                max_epochs_from_params = fixed_params.get('max_epochs')
                if max_epochs_from_params:
                    scheduler_constructor_kwargs['T_max'] = max_epochs_from_params
                else:
                    logger.warning("CosineAnnealingLR policy specified but T_max is not set and "
                                   "cannot be inferred from max_epochs. Scheduler might fail or use skorch default.")

        if actual_scheduler_class != torch.optim.lr_scheduler.ReduceLROnPlateau and 'monitor' in direct_lr_scheduler_args:
            logger.debug(
                f"Scheduler {actual_scheduler_class.__name__} does not use 'monitor'. Removing it from LRScheduler args.")
            direct_lr_scheduler_args.pop('monitor')

        lr_scheduler_instance = LRScheduler(
            policy=actual_scheduler_class,
            **direct_lr_scheduler_args,
            **scheduler_constructor_kwargs
        )

        processed_params[scheduler_key_prefix.strip('_')] = lr_scheduler_instance
        logger.debug(f"Created LRScheduler instance: policy={actual_scheduler_class.__name__}, "
                     f"direct_args={direct_lr_scheduler_args}, constructor_kwargs={scheduler_constructor_kwargs}")

    return processed_params


def expand_hyperparameter_grid(input_grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    processed_grid = input_grid.copy()

    if 'optimizer' in processed_grid:
        processed_grid['optimizer'] = _resolve_optimizer(processed_grid['optimizer'])

    scheduler_key_prefix = f'callbacks__{DEFAULT_LR_SCHEDULER_NAME}__'
    policy_key = f'{scheduler_key_prefix}policy'

    if policy_key in processed_grid:
        scheduler_policies_from_grid = processed_grid.pop(policy_key)
        if not isinstance(scheduler_policies_from_grid, list):
            scheduler_policies_from_grid = [scheduler_policies_from_grid]

        direct_lr_scheduler_params_grid: Dict[str, List[Any]] = {}
        pytorch_scheduler_param_grid: Dict[str, List[Any]] = {}

        for key in list(processed_grid.keys()):
            if key.startswith(scheduler_key_prefix):
                param_name = key.split('__')[-1]
                if param_name != 'policy':
                    if param_name in ['monitor', 'event_name', 'step_every']:
                        direct_lr_scheduler_params_grid[param_name] = processed_grid.pop(key)
                    else:
                        pytorch_scheduler_param_grid[param_name] = processed_grid.pop(key)

        generated_lr_schedulers = []

        for policy_name_or_class_item in scheduler_policies_from_grid:
            try:
                actual_scheduler_class_for_grid = _resolve_scheduler_policy_class(policy_name_or_class_item)
            except ValueError as e:
                logger.error(
                    f"Error resolving scheduler policy '{policy_name_or_class_item}' in grid: {e}. Skipping this policy.")
                continue

            current_policy_params_to_combine = {}
            current_policy_params_to_combine.update(direct_lr_scheduler_params_grid)
            current_policy_params_to_combine.update(pytorch_scheduler_param_grid)

            param_names = list(current_policy_params_to_combine.keys())
            param_value_lists = [current_policy_params_to_combine[name] for name in param_names]

            if not param_value_lists:
                param_value_lists_for_product = [[]]
                param_names_for_product = []
            else:
                param_value_lists_for_product = param_value_lists
                param_names_for_product = param_names

            for specific_param_combination_values in itertools.product(*param_value_lists_for_product):
                scheduler_instance_kwargs = dict(zip(param_names_for_product, specific_param_combination_values))

                lr_scheduler_direct_args_grid = {}
                torch_scheduler_constructor_kwargs_grid = {}

                for k_arg, v_arg in scheduler_instance_kwargs.items():
                    if k_arg in ['monitor', 'event_name', 'step_every']:
                        lr_scheduler_direct_args_grid[k_arg] = v_arg
                    else:
                        torch_scheduler_constructor_kwargs_grid[k_arg] = v_arg

                if actual_scheduler_class_for_grid != torch.optim.lr_scheduler.ReduceLROnPlateau and 'monitor' in lr_scheduler_direct_args_grid:
                    lr_scheduler_direct_args_grid.pop('monitor')

                generated_lr_schedulers.append(
                    LRScheduler(policy=actual_scheduler_class_for_grid,
                                **lr_scheduler_direct_args_grid,
                                **torch_scheduler_constructor_kwargs_grid)
                )

        if generated_lr_schedulers:
            processed_grid[scheduler_key_prefix.strip('_')] = generated_lr_schedulers
        elif direct_lr_scheduler_params_grid or pytorch_scheduler_param_grid:
            logger.warning(
                f"LR Scheduler policies {scheduler_policies_from_grid} were specified, but no valid parameter combinations generated LRScheduler objects.")

    return processed_grid
