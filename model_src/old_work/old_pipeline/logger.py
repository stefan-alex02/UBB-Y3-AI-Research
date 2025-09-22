# import logging
# from datetime import datetime
# from pathlib import Path
# import sys
#
# def create_logger(log_dir: str = 'logs') -> logging.Logger:
#     """
#     Creates a logger that logs messages to both a file and the console.
#     :param log_dir: Directory to store log files (in UTF-8 encoding). Name of the log file will be
#                     'log_<timestamp>.txt'.
#     :return: Logger object
#     """
#
#     # Properly join paths using Path objects
#     path = Path(log_dir).resolve()
#
#     # Create logs directory inside results directory if it doesn't exist
#     path.mkdir(parents=True, exist_ok=True)
#
#     # Create timestamp for unique log file
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     log_filename = Path(path) / f'log_{timestamp}.txt'
#
#     # Create formatters
#     file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
#                                        datefmt='%Y-%m-%d %H:%M:%S')
#     console_formatter = logging.Formatter('%(asctime)s | %(message)s',
#                                           datefmt='%H:%M:%S')
#
#     # Create and configure file handler (that supports UTF-8)
#     file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(file_formatter)
#
#     # Create and configure console handler
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(console_formatter)
#
#     # Get logger and add handlers
#     logger = logging.getLogger('logger')
#     logger.setLevel(logging.INFO)
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
#
#     return logger