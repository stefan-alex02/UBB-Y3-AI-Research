import asyncio
import functools
import logging
from typing import Callable, Tuple, Dict, Any

# Use your consistent FastAPI app logger name
from ..core.config import settings, \
    APP_LOGGER_NAME  # Assuming APP_FASTAPI_LOGGER_NAME is in settings or defined globally
logger = logging.getLogger(APP_LOGGER_NAME)

# Simple in-memory queue
# For production with multiple workers or persistence, use Redis+Celery or similar.
_experiment_queue = asyncio.Queue()
_task_processor_active = False


async def add_experiment_to_queue(task_func: Callable, args: Tuple = (), kwargs: Dict = None):
    if kwargs is None:
        kwargs = {}
    # We still store args and kwargs separately in the queue item
    await _experiment_queue.put((task_func, args, kwargs))
    logger.info(f"Experiment task {task_func.__name__} added to queue. Queue size: {_experiment_queue.qsize()}")
    ensure_processor_is_running()  # ensure_processor_is_running should be async if it creates tasks


async def _process_queue():  # Make sure this is async
    global _task_processor_active
    _task_processor_active = True
    logger.info("Experiment task processor started.")
    while True:
        try:
            task_func, pos_args, kw_args = await _experiment_queue.get()  # Get func, positional args, and keyword args

            exp_id_for_log = kw_args.get('experiment_run_id', 'Unknown ID')  # Get ID for logging
            logger.info(f"Processing experiment {exp_id_for_log} from queue. Remaining: {_experiment_queue.qsize()}")

            try:
                loop = asyncio.get_event_loop()

                # Create a new function with arguments "baked in" using functools.partial
                # task_func is _execute_pipeline_task
                # kw_args is {"executor_params": ..., "experiment_run_id": ..., "artifact_repo": ...}
                # pos_args is currently an empty tuple ()

                # Ensure all expected kwargs for _execute_pipeline_task are present in kw_args
                # and that _execute_pipeline_task doesn't expect any positional args if pos_args is empty.

                # functools.partial creates a new callable that will be invoked with no further arguments
                # by run_in_executor, but will call task_func with the specified args and kwargs.
                # Example: if task_func expects (a, b, c=None)
                # partial_func = functools.partial(task_func, 1, b=2, c=3)
                # loop.run_in_executor(None, partial_func) -> effectively runs task_func(1, b=2, c=3)

                # Since _execute_pipeline_task takes only keyword arguments (based on its definition),
                # and we are passing an empty tuple for positional args:
                if pos_args:  # If there were any positional args
                    # This case needs careful handling if your task_func expects both
                    # For now, assuming _execute_pipeline_task primarily uses kwargs from the dict
                    logger.warning(
                        f"Positional arguments {pos_args} were provided but _execute_pipeline_task primarily uses keyword arguments. Ensure this is intended.")
                    # If task_func actually uses these, you'd do:
                    # target_for_executor = functools.partial(task_func, *pos_args, **kw_args)
                else:
                    target_for_executor = functools.partial(task_func, **kw_args)

                await loop.run_in_executor(None, target_for_executor)

                logger.info(f"Finished processing experiment {exp_id_for_log}.")
            except Exception as e:
                experiment_id_for_log_inner = kw_args.get('experiment_run_id', 'Unknown ID inner')
                logger.error(f"Error processing experiment {experiment_id_for_log_inner} from queue: {e}",
                             exc_info=True)
            finally:
                _experiment_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Experiment task processor cancelled.")
            _task_processor_active = False
            break
        except Exception as e:
            logger.critical(f"Critical error in task processor loop: {e}", exc_info=True)
            # Potentially re-initialize or stop, depending on error severity
            await asyncio.sleep(5) # Avoid rapid looping on persistent error

def ensure_processor_is_running():
    """Ensures the queue processor task is running in the asyncio event loop."""
    global _task_processor_active
    if not _task_processor_active:
        # This should ideally be started once when the FastAPI app starts,
        # but this provides a way to kick it off if it somehow stops or isn't started.
        # A better place for initial start is in FastAPI's lifespan event.
        logger.info("Task processor not active. Attempting to start.")
        asyncio.create_task(_process_queue())

# You would call ensure_processor_is_running() once during FastAPI startup
# in your main.py lifespan function.