import torch
import itertools
from typing import Dict, List, Any, Union, Type, Optional
from skorch.callbacks import LRScheduler  # Important import

from model_src.server.ml.logger_utils import logger

# Optimizer mapping (could be extended)
OPTIMIZER_MAP: Dict[str, Type[torch.optim.Optimizer]] = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

# Default LRScheduler name (must match the name in get_default_callbacks)
DEFAULT_LR_SCHEDULER_NAME = 'default_lr_scheduler'


def _resolve_optimizer(opt_val_or_list: Any) -> Any:
    """Converts optimizer string(s) to type(s)."""
    if isinstance(opt_val_or_list, list):
        return [OPTIMIZER_MAP.get(opt.lower(), opt) if isinstance(opt, str) else opt for opt in opt_val_or_list]
    elif isinstance(opt_val_or_list, str):
        return OPTIMIZER_MAP.get(opt_val_or_list.lower(), opt_val_or_list)
    return opt_val_or_list


def parse_fixed_hyperparameters(
        fixed_params: Dict[str, Any],
        default_max_epochs_for_cosine: Optional[int] = None  # For CosineAnnealingLR T_max
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

    # 1. Resolve Optimizer
    if 'optimizer' in processed_params:
        opt_val = processed_params['optimizer']
        if isinstance(opt_val, str):
            opt_type = OPTIMIZER_MAP.get(opt_val.lower())
            if opt_type is None:
                raise ValueError(f"Unsupported optimizer string in fixed_params: '{opt_val}'")
            processed_params['optimizer'] = opt_type
        elif not (isinstance(opt_val, type) and issubclass(opt_val, torch.optim.Optimizer)):
            raise TypeError(f"Optimizer in fixed_params must be a string or Optimizer type, got {type(opt_val)}")

    # 2. Construct LRScheduler Callback Object if specified
    scheduler_key_prefix = f'callbacks__{DEFAULT_LR_SCHEDULER_NAME}__'
    policy_key = f'{scheduler_key_prefix}policy'

    if policy_key in processed_params:
        policy_name_or_type = processed_params.pop(policy_key)
        policy_str = policy_name_or_type if isinstance(policy_name_or_type, str) else policy_name_or_type.__name__

        scheduler_constructor_kwargs: Dict[str, Any] = {}
        direct_lr_scheduler_args: Dict[str, Any] = {}  # For LRScheduler's own params like 'monitor'

        # Collect all params for this scheduler
        param_keys_for_this_scheduler = [
            k for k in processed_params if k.startswith(scheduler_key_prefix)
        ]

        for key in param_keys_for_this_scheduler:
            param_name = key.split('__')[-1]
            # Check if it's a direct LRScheduler argument or for the torch scheduler
            if param_name in ['monitor', 'event_name', 'step_every']:  # Direct LRScheduler args
                direct_lr_scheduler_args[param_name] = processed_params.pop(key)
            else:  # Kwarg for the torch.optim.lr_scheduler
                scheduler_constructor_kwargs[param_name] = processed_params.pop(key)

        # Handle T_max for CosineAnnealingLR specifically if not provided
        if policy_str == 'CosineAnnealingLR' and 'T_max' not in scheduler_constructor_kwargs:
            if default_max_epochs_for_cosine:
                scheduler_constructor_kwargs['T_max'] = default_max_epochs_for_cosine
            else:  # Try to get from fixed_params if available, else warning
                max_epochs_from_params = fixed_params.get('max_epochs')
                if max_epochs_from_params:
                    scheduler_constructor_kwargs['T_max'] = max_epochs_from_params
                else:
                    logger.warning("CosineAnnealingLR policy specified but T_max is not set and "
                                   "cannot be inferred from max_epochs. Scheduler might fail or use skorch default.")

        # Instantiate the LRScheduler object
        # Note: get_default_callbacks also has logic for setting verbose=False etc.
        # For simplicity here, if verbose is not in scheduler_constructor_kwargs, it will use torch default.
        # You could merge with defaults from get_default_callbacks's internal logic if desired.
        lr_scheduler_instance = LRScheduler(
            policy=policy_str,
            **direct_lr_scheduler_args,  # monitor, etc.
            **scheduler_constructor_kwargs  # factor, patience, T_max, step_size, etc.
        )

        # Replace the individual scheduler params with the actual LRScheduler object
        # The key for the callback object itself is 'callbacks__<name>'
        processed_params[scheduler_key_prefix.strip('_')] = lr_scheduler_instance

    # Process other callbacks if needed (e.g., EarlyStopping patience)
    # Example:
    # if 'callbacks__default_early_stopping__patience' in processed_params:
    #     # This is fine, Skorch will handle it directly if it's a simple attribute.
    #     # If EarlyStopping itself needed to be replaced with a custom instance,
    #     # similar logic to LRScheduler would apply.
    #     pass

    return processed_params


def expand_hyperparameter_grid(input_grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Expands a user-friendly hyperparameter grid to one directly usable by
    GridSearchCV with Skorch, especially for LRSchedulers.

    Args:
        input_grid: A dictionary where keys are parameter names and values are lists
                    of options to try.
                    - 'optimizer' can be a list of optimizer name strings.
                    - 'callbacks__<name>__policy' can be a list of scheduler policy strings.
                    - Other 'callbacks__<name>__<scheduler_param>' are parameters for those policies.

    Returns:
        A new grid dictionary where optimizer strings are resolved to types, and
        LR scheduler configurations are expanded into a list of LRScheduler objects
        for the 'callbacks__<name>' key.
    """
    processed_grid = input_grid.copy()

    # 1. Resolve Optimizer strings to types
    if 'optimizer' in processed_grid:
        processed_grid['optimizer'] = _resolve_optimizer(processed_grid['optimizer'])

    # 2. Handle LR Scheduler expansion
    scheduler_key_prefix = f'callbacks__{DEFAULT_LR_SCHEDULER_NAME}__'
    policy_key = f'{scheduler_key_prefix}policy'

    if policy_key in processed_grid:
        scheduler_policies = processed_grid.pop(policy_key)  # Remove policy key
        if not isinstance(scheduler_policies, list):
            scheduler_policies = [scheduler_policies]

        # Collect all other parameters for this scheduler
        scheduler_specific_param_grid: Dict[str, List[Any]] = {}
        keys_to_remove_from_main_grid = []

        # Direct LRScheduler parameters (like 'monitor')
        direct_lr_scheduler_params = {}
        for key in list(processed_grid.keys()):  # Iterate over copy of keys
            if key.startswith(scheduler_key_prefix) and not key.startswith(f'{scheduler_key_prefix}fn_kwargs__'):
                # This is a direct param for LRScheduler itself (e.g., monitor)
                # or a param for the underlying torch scheduler if not 'policy'
                param_name = key.split('__')[-1]
                if param_name != 'policy':  # policy already handled
                    direct_lr_scheduler_params[param_name] = processed_grid.pop(key)
                    keys_to_remove_from_main_grid.append(key)

        # Parameters for the PyTorch scheduler (previously under fn_kwargs)
        # Now we expect them directly like 'callbacks__default_lr_scheduler__factor'
        # These will be collected and passed as **kwargs to LRScheduler
        pytorch_scheduler_param_grid: Dict[str, List[Any]] = {}
        for key in list(processed_grid.keys()):
            if key.startswith(scheduler_key_prefix):
                # If it was not a direct LRScheduler param (like monitor), it's for the torch scheduler
                if key.split('__')[-1] not in ['monitor', 'event_name', 'step_every',
                                               'policy']:  # Known LRScheduler direct params
                    param_name = key.split('__')[-1]
                    pytorch_scheduler_param_grid[param_name] = processed_grid.pop(key)
                    keys_to_remove_from_main_grid.append(key)

        generated_lr_schedulers = []
        for policy_name_or_type in scheduler_policies:
            policy_str = policy_name_or_type if isinstance(policy_name_or_type, str) else policy_name_or_type.__name__

            # Create combinations of this policy's specific parameters
            current_policy_params_to_combine = {}
            # Add direct LRScheduler params if they are defined for this grid point
            for k, v_list in direct_lr_scheduler_params.items():
                current_policy_params_to_combine[k] = v_list

            # Add PyTorch scheduler specific params
            for k, v_list in pytorch_scheduler_param_grid.items():
                current_policy_params_to_combine[k] = v_list

            # Generate all combinations of parameters for this specific policy
            param_names = list(current_policy_params_to_combine.keys())
            param_value_lists = [current_policy_params_to_combine[name] for name in param_names]

            for specific_param_combination_values in itertools.product(*param_value_lists):
                scheduler_instance_kwargs = dict(zip(param_names, specific_param_combination_values))

                # Separate LRScheduler direct args from torch scheduler kwargs
                lr_scheduler_direct_args = {}
                torch_scheduler_constructor_kwargs = {}

                if 'monitor' in scheduler_instance_kwargs:
                    lr_scheduler_direct_args['monitor'] = scheduler_instance_kwargs.pop('monitor')
                if 'event_name' in scheduler_instance_kwargs:
                    lr_scheduler_direct_args['event_name'] = scheduler_instance_kwargs.pop('event_name')
                if 'step_every' in scheduler_instance_kwargs:
                    lr_scheduler_direct_args['step_every'] = scheduler_instance_kwargs.pop('step_every')

                # Remaining kwargs are for the torch scheduler
                torch_scheduler_constructor_kwargs = scheduler_instance_kwargs

                generated_lr_schedulers.append(
                    LRScheduler(policy=policy_str, **lr_scheduler_direct_args, **torch_scheduler_constructor_kwargs)
                )

        if generated_lr_schedulers:
            processed_grid[
                scheduler_key_prefix.strip('_')] = generated_lr_schedulers  # e.g. 'callbacks__default_lr_scheduler'
        elif direct_lr_scheduler_params or pytorch_scheduler_param_grid:  # Policy might have been a direct type with no params in grid
            logger.warning(
                f"LR Scheduler policy {scheduler_policies} was specified, but no valid parameter combinations generated LRScheduler objects.")

    # Handle other callbacks like EarlyStopping if they are also tuned
    early_stopping_prefix = f'callbacks__default_early_stopping__'
    early_stopping_params_in_grid = {
        k.replace(early_stopping_prefix, ''): v
        for k, v in processed_grid.items() if k.startswith(early_stopping_prefix)
    }
    if early_stopping_params_in_grid:
        # If tuning EarlyStopping, it also needs to be a list of objects
        # This part gets complex quickly if multiple callbacks are tuned with multiple params each.
        # For now, assume EarlyStopping params are direct attributes of the ES object.
        # If 'callbacks__default_early_stopping__patience': [10, 15] is in the grid,
        # Skorch's default set_params for callbacks should handle this.
        # The "object replacement" strategy is mainly for when the *constructor signature*
        # of the underlying component changes significantly (like LRScheduler with different policies).
        pass  # Skorch should handle direct param setting for EarlyStopping

    return processed_grid
