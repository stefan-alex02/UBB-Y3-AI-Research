import logging
import sys
from pathlib import Path
from typing import Union

from .config import LogColors, logger_name_global


class EnhancedFormatter(logging.Formatter):
    debug_emoji = '🐞'
    info_emoji = 'ℹ️'
    warning_emoji = '⚠️'
    error_emoji = '❌'
    critical_emoji = '💀'
    default_emoji = '?'

    # Define widths for alignment
    filename_padding = 10
    funcname_padding = 18
    lineno_padding = 4
    # Calculate total width for Location column: Brackets + file + : + func + : + line
    total_location_width = 2 + filename_padding + 1 + funcname_padding + 1 + lineno_padding # = 43

    level_to_color = {
        logging.DEBUG: LogColors.CYAN, logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW, logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD + LogColors.RED,
    }
    level_to_emoji = {
        logging.DEBUG: debug_emoji, logging.INFO: info_emoji,
        logging.WARNING: warning_emoji, logging.ERROR: error_emoji,
        logging.CRITICAL: critical_emoji,
    }

    date_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, use_colors=True):
        # Basic init, we override format() completely
        super().__init__(fmt=None, datefmt=self.date_format)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # --- Start Manual Formatting ---
        try:
            # 1. Timestamp
            timestamp = self.formatTime(record, self.date_format)

            # 2. Level Name and Color
            level_name = record.levelname
            level_color = self.level_to_color.get(record.levelno, '') if self.use_colors else ''
            reset_color = LogColors.RESET if self.use_colors and level_color else ''
            formatted_level = f"{level_color}{level_name:<8}{reset_color}" # Pad to 8 chars

            # 3. Location String
            filename = getattr(record, 'filename', '?')
            funcname = getattr(record, 'funcName', '?')
            lineno = getattr(record, 'lineno', 0)

            # Truncate filename (without extension)
            if '.' in filename: filename = filename.rsplit('.', 1)[0]
            max_file_len = self.filename_padding
            truncated_filename = (filename[:max_file_len-3] + '...') if len(filename) > max_file_len else filename

            # Truncate funcName
            max_func_len = self.funcname_padding
            truncated_funcname = (funcname[:max_func_len-3] + '...') if len(funcname) > max_func_len else funcname

            # Format location with padding
            location_str = (
                f"[{truncated_filename:<{self.filename_padding}}:"
                f"{truncated_funcname:<{self.funcname_padding}}:"
                f"{lineno:>{self.lineno_padding}}]"
            )
            # Pad the entire location block
            formatted_location = f"{location_str:<{self.total_location_width}}"

            # 4. Emoji Prefix + Message
            emoji_prefix = self.level_to_emoji.get(record.levelno, self.default_emoji)
            message = record.getMessage() # Get the formatted message
            formatted_message = f"{emoji_prefix} {message}" # Add space after emoji

            # 5. Combine parts
            log_entry = f"{timestamp} | {formatted_level} | {formatted_location} | {formatted_message}"
            return log_entry

        except Exception as e:
            # Fallback formatting on any error during manual formatting
            record.msg = f"!!! LOG FORMATTING ERROR: {e}. Original message: {record.getMessage()}"
            # Use a basic formatter as ultimate fallback
            bf = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt=self.date_format)
            return bf.format(record)


def get_log_header(use_colors: bool = True) -> str:
    location_width = EnhancedFormatter.total_location_width # Use width from class
    # Remove Emoji column header
    header_title = f"{'Timestamp':<19} | {'Level':<8} | {'Location':<{location_width}} | {'Message'}"
    separator = f"{'-'*19}-+-{'-'*8}-+-{'-'*location_width}-+-{'-'*50}" # Removed emoji separator part
    if use_colors: separator = f"{LogColors.DIM}{separator}{LogColors.RESET}"
    return f"{header_title}\n{separator}"


def write_log_header_if_needed(log_path: Path):
    try:
        is_new_or_empty = not log_path.is_file() or log_path.stat().st_size == 0
        if is_new_or_empty:
            header = get_log_header(use_colors=False)
            with open(log_path, 'a', encoding='utf-8') as f: f.write(header + "\n")
            return True
    except Exception as e:
        print(f"Error writing log header to {log_path}: {e}", file=sys.stderr)
    return False


def setup_logger(name: str,
                 log_dir: Union[str, Path], # <<< Accept directory instead of full path
                 log_filename: str = 'classification.log', # <<< Default filename
                 level: int = logging.INFO,
                 use_colors: bool = True) -> logging.Logger:
    """Sets up the logger to log to console and a file within the specified directory."""
    _logger = logging.getLogger(name)
    # Prevent duplicate handlers if logger already exists (e.g., in interactive sessions)
    if _logger.hasHandlers():
        _logger.handlers.clear()
        # logger.info("Cleared existing logger handlers.") # Optional debug log

    _logger.setLevel(level)
    _logger.propagate = False # Prevent propagation to root logger

    # Console Handler (always add)
    console_formatter = EnhancedFormatter(use_colors=use_colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    _logger.addHandler(console_handler)

    # File Handler (if directory provided)
    if log_dir:
        log_path = Path(log_dir) / log_filename
        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Check and write header if needed
        is_new_file = write_log_header_if_needed(log_path)
        try:
            file_formatter = EnhancedFormatter(use_colors=False)
            # Use 'a' mode to append to the log file
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            _logger.addHandler(file_handler)
            # Log initialization message only if it's a new file or handlers were just added
            print(get_log_header(use_colors=True)) # Print header after setup
            if is_new_file or not _logger.handlers: # Simple check if handlers were just added
                 _logger.info(f"Logger '{name}' initialized. Log file: {log_path}")

        except Exception as e:
             # Use logger AFTER console handler is added
             _logger.error(f"Failed to create file handler for {log_path}: {e}")
    else:
        _logger.warning("No log directory provided. Logging to console only.")

    return _logger

# This logger will be configured by PipelineExecutor
logger = logging.getLogger(logger_name_global)
