import asyncio
import logging
from typing import Callable, Any
from fastapi import BackgroundTasks # If using FastAPI's simple background tasks

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
        # If task_func is synchronous, run it in a thread pool executor
        # For simplicity, if PipelineExecutor.run() is synchronous and long-running,
        # this approach using asyncio.to_thread (Python 3.9+) or loop.run_in_executor is needed.
        # If PipelineExecutor.run() itself can be made async, that's even better.

        # Assuming PipelineExecutor.run() is synchronous
        loop.run_in_executor(None, task_func, *args, **kwargs)
        # The loop.run_in_executor submits the task and returns immediately.
        # The actual execution happens in a separate thread managed by the executor.
        # We don't await its completion here as this function is for fire-and-forget.
        # The task_func itself should handle callbacks to Java.
    except Exception as e:
        logger.error(f"Error starting background task {task_func.__name__}: {e}", exc_info=True)
    # finally: # Loop management can be tricky here, usually FastAPI handles it better
    #     loop.close() # Closing the loop here might be problematic if other tasks are scheduled on it
