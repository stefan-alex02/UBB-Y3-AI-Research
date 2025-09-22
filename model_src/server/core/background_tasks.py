import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger("fastapi_app")

def run_in_background(task_func: Callable, *args: Any, **kwargs: Any):
    """
    Runs a function in a separate thread to avoid blocking the main FastAPI event loop.
    For more robust production use, consider Celery or similar.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logger.info(f"Starting background task: {task_func.__name__} with args: {args}, kwargs: {kwargs}")

        loop.run_in_executor(None, task_func, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error starting background task {task_func.__name__}: {e}", exc_info=True)
