import asyncio
import functools
import logging
from typing import Callable, Tuple, Dict

from ..core.config import APP_LOGGER_NAME

logger = logging.getLogger(APP_LOGGER_NAME)

_experiment_queue = asyncio.Queue()
_task_processor_active = False


async def add_experiment_to_queue(task_func: Callable, args: Tuple = (), kwargs: Dict = None):
    if kwargs is None:
        kwargs = {}
    await _experiment_queue.put((task_func, args, kwargs))
    logger.info(f"Experiment task {task_func.__name__} added to queue. Queue size: {_experiment_queue.qsize()}")
    ensure_processor_is_running()


async def _process_queue():
    global _task_processor_active
    _task_processor_active = True
    logger.info("Experiment task processor started.")
    while True:
        try:
            task_func, pos_args, kw_args = await _experiment_queue.get()

            exp_id_for_log = kw_args.get('experiment_run_id', 'Unknown ID')  # Get ID for logging
            logger.info(f"Processing experiment {exp_id_for_log} from queue. Remaining: {_experiment_queue.qsize()}")

            try:
                loop = asyncio.get_event_loop()
                if pos_args:
                    logger.warning(
                        f"Positional arguments {pos_args} were provided but _execute_pipeline_task primarily uses keyword arguments. Ensure this is intended.")
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
            await asyncio.sleep(5)

def ensure_processor_is_running():
    """Ensures the queue processor task is running in the asyncio event loop."""
    global _task_processor_active
    if not _task_processor_active:
        logger.info("Task processor not active. Attempting to start.")
        asyncio.create_task(_process_queue())
