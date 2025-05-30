import logging
from numbers import Number
from typing import List, Tuple, Optional

from skorch.callbacks import Callback
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring
from skorch.callbacks import ProgressBar

from ..config import logger_name_global
from ..logger_utils import logger

try:
    import tabulate
except ImportError:
    tabulate = None


class FileLogTable(Callback):
    # Class attributes for configuration
    _COLUMNS = ['epoch', 'train_acc', 'train_loss', 'valid_acc', 'valid_loss', 'cp', 'lr', 'dur']
    _COLUMN_FORMATS = {
        'epoch': '{}', 'train_acc': '{:.4f}', 'train_loss': '{:.4f}',
        'valid_acc': '{:.4f}', 'valid_loss': '{:.4f}',
        'cp': '{}', 'lr': '{:.4e}', 'dur': '{:.4f}',
    }
    _COLALIGN = ("right", "right", "right", "right", "right", "center", "right", "right")
    _tabulate_warning_shown = False # Class attribute for one-time warning

    # ASCII indicators for improvements
    _IMPROVEMENT_LOSS_BETTER = '▼ ' # Loss went down
    _IMPROVEMENT_ACC_BETTER = '▲ '  # Accuracy went up
    _NO_IMPROVEMENT_SYMBOL = '  ' # Space if no improvement or not best

    def __init__(self, logger_instance=None):
        self.logger_instance = logger_instance or logger
        # DO NOT initialize instance state like _headers_printed_this_fit here.
        # It will be handled by initialize() or on_train_begin().

    def initialize(self): # Called by Skorch for each instance before a fit
        super().initialize()
        # Initialize/reset instance-specific state FOR THIS FIT
        self._headers_printed_this_fit: bool = False
        self._current_fit_epoch_count: int = 0
        self._best_train_loss: float = float('inf')
        self._best_train_acc: float = float('-inf')
        self._best_valid_loss: float = float('inf')
        self._best_valid_acc: float = float('-inf')

        # self.logger_instance.debug(f"FileLogTable Initialized/Reset for fit: {hex(id(self))}")

        if tabulate is None and not FileLogTable._tabulate_warning_shown:
             # Use the global logger for this warning
             logging.getLogger(logger_name_global).warning(
                 "Package 'tabulate' is not installed. FileLogTable will not print epoch summaries."
             )
             FileLogTable._tabulate_warning_shown = True
        return self

    # on_train_begin is also a good place for state reset per fit,
    # initialize() is called when the net is initialized, on_train_begin when fit starts.
    # For cloned callbacks in GridSearchCV, initialize() should be called for each clone.
    # Let's rely on initialize() for resetting these per-fit states.

    def on_epoch_end(self, net, **kwargs):
        # self.logger_instance.debug(f"FileLogTable on_epoch_end: inst={hex(id(self))}, header_printed={getattr(self, '_headers_printed_this_fit', 'NotSet')}, epoch_count={getattr(self, '_current_fit_epoch_count', 'NotSet')}")
        history = net.history
        if not history or tabulate is None: return
        last_epoch_data = history[-1]
        if not last_epoch_data: return

        # Ensure state attributes are initialized if initialize somehow wasn't called (defensive)
        if not hasattr(self, '_current_fit_epoch_count'): self.initialize()

        self._current_fit_epoch_count += 1

        # --- Update bests for emoji logic ---
        current_epoch_is_best = {}
        current_train_loss = last_epoch_data.get('train_loss', float('inf'))
        if current_train_loss < self._best_train_loss:
            self._best_train_loss = current_train_loss; current_epoch_is_best['train_loss'] = True
        else: current_epoch_is_best['train_loss'] = (current_train_loss == self._best_train_loss and current_train_loss != float('inf'))

        current_train_acc = last_epoch_data.get('train_acc', float('-inf'))
        if current_train_acc > self._best_train_acc:
            self._best_train_acc = current_train_acc; current_epoch_is_best['train_acc'] = True
        else: current_epoch_is_best['train_acc'] = (current_train_acc == self._best_train_acc and current_train_acc != float('-inf'))

        current_valid_loss = last_epoch_data.get('valid_loss', float('inf'))
        if current_valid_loss < self._best_valid_loss:
            self._best_valid_loss = current_valid_loss; current_epoch_is_best['valid_loss'] = True
        else: current_epoch_is_best['valid_loss'] = (current_valid_loss == self._best_valid_loss and current_valid_loss != float('inf'))

        current_valid_acc = last_epoch_data.get('valid_acc', float('-inf'))
        if current_valid_acc > self._best_valid_acc:
            self._best_valid_acc = current_valid_acc; current_epoch_is_best['valid_acc'] = True
        else: current_epoch_is_best['valid_acc'] = (current_valid_acc == self._best_valid_acc and current_valid_acc != float('-inf'))
        # --- End update bests ---

        row_values = []
        for key in FileLogTable._COLUMNS: # Use class attribute
            val_str = ""
            value = last_epoch_data.get(key)
            emoji_prefix = FileLogTable._NO_IMPROVEMENT_SYMBOL

            if key == 'cp':
                val_str = '+' if last_epoch_data.get('event_cp') is True else ('✓' if isinstance(last_epoch_data.get('event_cp'), str) else '')
            elif key == 'lr':
                lr_val = value
                if lr_val is None and hasattr(net, 'optimizer_') and net.optimizer_ and net.optimizer_.param_groups:
                    try: lr_val = net.optimizer_.param_groups[0]['lr']
                    except: lr_val = float('nan')
                val_str = FileLogTable._COLUMN_FORMATS['lr'].format(lr_val) if isinstance(lr_val, Number) else "N/A"
            elif value is not None and isinstance(value, Number):
                fmt = FileLogTable._COLUMN_FORMATS.get(key, '{}')
                try:
                    val_str = fmt.format(value)
                except:
                    val_str = str(value)

                indicator = FileLogTable._NO_IMPROVEMENT_SYMBOL  # Default to space
                if key == 'train_loss' and current_epoch_is_best.get('train_loss'):
                    indicator = FileLogTable._IMPROVEMENT_LOSS_BETTER
                elif key == 'train_acc' and current_epoch_is_best.get('train_acc'):
                    indicator = FileLogTable._IMPROVEMENT_ACC_BETTER
                elif key == 'valid_loss' and current_epoch_is_best.get('valid_loss'):
                    indicator = FileLogTable._IMPROVEMENT_LOSS_BETTER
                elif key == 'valid_acc' and current_epoch_is_best.get('valid_acc'):
                    indicator = FileLogTable._IMPROVEMENT_ACC_BETTER

                val_str = f"{indicator}{val_str}"
            elif value is not None: val_str = str(value)
            else: val_str = ""
            row_values.append(val_str)

        if not row_values: return

        table_data = [row_values]
        try:
            if not self._headers_printed_this_fit:
                table_str = tabulate.tabulate(table_data, headers=FileLogTable._COLUMNS, tablefmt="simple", colalign=FileLogTable._COLALIGN)
                self._headers_printed_this_fit = True
            else:
                full_row_table_str = tabulate.tabulate(table_data, headers=FileLogTable._COLUMNS, tablefmt="simple", colalign=FileLogTable._COLALIGN)
                table_str = full_row_table_str.splitlines()[-1]

            for line in table_str.splitlines():
                self.logger_instance.info(line)

        except Exception as e:
            self.logger_instance.error(f"FileLogTable: Error generating table for epoch {last_epoch_data.get('epoch', '?')}: {e}", exc_info=True)
            raw_log = f"Epoch {last_epoch_data.get('epoch', '?')} Data (tabulate failed): " + ", ".join(f"{h}={v}" for h,v in zip(FileLogTable._COLUMNS, row_values))
            self.logger_instance.info(raw_log)


# TODO use a custom sampler for grid search to allow interval sampling
# or make the adapter use the params to create a customized LR scheduler
def get_default_callbacks(
    early_stopping_monitor: str = 'valid_loss',
    early_stopping_patience: int = 10,
    lr_scheduler_policy: str = 'ReduceLROnPlateau', # Default policy from Pipeline __init__
    lr_scheduler_monitor: Optional[str] = 'valid_loss',
    **kwargs # Renaming for clarity, these are constructor args for the PyTorch scheduler
) -> List[Tuple[str, Callback]]:
    is_loss_metric = early_stopping_monitor.endswith('_loss')

    # These kwargs are for the underlying torch.optim.lr_scheduler.* constructor
    # If not provided by the caller (ClassificationPipeline.__init__),
    # we can set some sensible internal defaults here for the *default policy*.
    scheduler_kwargs_to_use = kwargs.copy() # Start with what's passed

    if lr_scheduler_policy == 'ReduceLROnPlateau':
        scheduler_kwargs_to_use.setdefault('mode', 'min' if lr_scheduler_monitor and lr_scheduler_monitor.endswith('_loss') else 'max')
        scheduler_kwargs_to_use.setdefault('factor', 0.1)
        scheduler_kwargs_to_use.setdefault('patience', 5)
        scheduler_kwargs_to_use.setdefault('min_lr', 1e-6)
        scheduler_kwargs_to_use.setdefault('verbose', False)
    elif lr_scheduler_policy == 'StepLR':
        scheduler_kwargs_to_use.setdefault('step_size', 10)
        scheduler_kwargs_to_use.setdefault('gamma', 0.1)
        scheduler_kwargs_to_use.setdefault('verbose', False)
    elif lr_scheduler_policy == 'CosineAnnealingLR':
        # T_max is critical and often context-dependent (e.g., max_epochs).
        # If not in scheduler_kwargs_to_use, LRScheduler might infer it or error.
        scheduler_kwargs_to_use.setdefault('eta_min', 0)
        scheduler_kwargs_to_use.setdefault('verbose', False)
        if 'T_max' not in scheduler_kwargs_to_use:
            logger.warning(f"CosineAnnealingLR selected as default policy, but T_max not in default_scheduler_fn_kwargs. "
                           f"Ensure T_max is set via 'callbacks__default_lr_scheduler__T_max' or it might default to max_epochs.")
    # Add other policies and their sensible base defaults if not overridden by kwargs

    lr_scheduler_instance = LRScheduler(
        policy=lr_scheduler_policy,
        monitor=lr_scheduler_monitor if lr_scheduler_policy == 'ReduceLROnPlateau' else None,
        **scheduler_kwargs_to_use # Pass the final composed kwargs
    )
    logger.debug(f"Default LRScheduler callback instance created with policy: {lr_scheduler_policy}, "
                 f"kwargs: {scheduler_kwargs_to_use}, "
                 f"monitor: {lr_scheduler_monitor if lr_scheduler_policy == 'ReduceLROnPlateau' else 'N/A'}")

    callbacks_list = [
        ('progress_bar', ProgressBar()),
        ('default_early_stopping', EarlyStopping(
            monitor=early_stopping_monitor, patience=early_stopping_patience,
            load_best=True, lower_is_better=is_loss_metric
        )),
        ('default_lr_scheduler', lr_scheduler_instance),
        ('default_train_acc_scorer', EpochScoring(
            scoring='accuracy', lower_is_better=False, on_train=True, name='train_acc'
        )),
        ('file_log_table_cb', FileLogTable())
    ]
    return callbacks_list
