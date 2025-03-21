"""
Logging utilities for the Agentis MCP framework.
"""

import logging
import sys
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


# Global logger configuration
_loggers: Dict[str, logging.Logger] = {}
_console = Console(stderr=True)
_log_level = logging.INFO
_log_format = "%(message)s"
_log_handlers = [RichHandler(console=_console, rich_tracebacks=True)]


def configure_logging(level: int = logging.INFO, add_file_handler: Optional[str] = None) -> None:
    """
    Configure the logging system.
    
    Args:
        level: Logging level.
        add_file_handler: If provided, also log to this file.
    """
    global _log_level, _log_handlers
    
    _log_level = level
    _log_handlers = [RichHandler(console=_console, rich_tracebacks=True)]
    
    if add_file_handler:
        file_handler = logging.FileHandler(add_file_handler)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        _log_handlers.append(file_handler)
    
    # Update existing loggers
    for logger in _loggers.values():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        for handler in _log_handlers:
            logger.addHandler(handler)
        
        logger.setLevel(_log_level)


class PatchedLogger(logging.Logger):
    """
    A logger that supports the 'data' parameter and other keyword arguments.
    """
    
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
    
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, data=None, **kwargs):
        """
        Low-level logging method. Support for 'data' parameter.
        """
        # If data is provided, add it to the message
        if data is not None:
            if args:
                args = args + (data,)
            else:
                msg = f"{msg} {data}"
        
        # Call the parent _log method
        super()._log(level, msg, args, exc_info, extra, stack_info)


# Register our custom logger class
logging.setLoggerClass(PatchedLogger)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(_log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our handlers
    for handler in _log_handlers:
        logger.addHandler(handler)
    
    _loggers[name] = logger
    return logger


class Logger:
    """
    Enhanced logger with structured logging capabilities.
    """
    
    def __init__(self, name: str):
        self._logger = get_logger(name)
    
    def debug(self, message: str, data: Any = None, **kwargs) -> None:
        """Log a debug message with optional structured data."""
        if data:
            self._logger.debug(f"{message} {data}")
        else:
            self._logger.debug(message)
    
    def info(self, message: str, data: Any = None, **kwargs) -> None:
        """Log an info message with optional structured data."""
        if data:
            self._logger.info(f"{message} {data}")
        else:
            self._logger.info(message)
    
    def warning(self, message: str, data: Any = None, **kwargs) -> None:
        """Log a warning message with optional structured data."""
        if data:
            self._logger.warning(f"{message} {data}")
        else:
            self._logger.warning(message)
    
    def error(self, message: str, data: Any = None, exc_info: bool = False, **kwargs) -> None:
        """Log an error message with optional structured data and exception info."""
        if data:
            self._logger.error(f"{message} {data}", exc_info=exc_info)
        else:
            self._logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, data: Any = None, exc_info: bool = True, **kwargs) -> None:
        """Log a critical message with optional structured data and exception info."""
        if data:
            self._logger.critical(f"{message} {data}", exc_info=exc_info)
        else:
            self._logger.critical(message, exc_info=exc_info)
    
    def event(self, level: str, event_type: str, message: str, exc: Optional[Exception] = None, data: Dict = None, **kwargs) -> None:
        """
        Log a structured event.
        
        Args:
            level: Log level ("debug", "info", "warning", "error", "critical").
            event_type: Type of event.
            message: Event message.
            exc: Optional exception.
            data: Optional structured data.
        """
        if level == "debug":
            self._logger.debug(f"[{event_type}] {message}", exc_info=exc)
        elif level == "info":
            self._logger.info(f"[{event_type}] {message}", exc_info=exc)
        elif level == "warning":
            self._logger.warning(f"[{event_type}] {message}", exc_info=exc)
        elif level == "error":
            self._logger.error(f"[{event_type}] {message}", exc_info=exc)
        elif level == "critical":
            self._logger.critical(f"[{event_type}] {message}", exc_info=exc)
        else:
            self._logger.info(f"[{event_type}] {message}", exc_info=exc)
            
    # Add a generic _log method to handle any unexpected kwargs
    def _log(self, level: str, message: str, **kwargs) -> None:
        """
        Generic logging method that accepts any keyword arguments.
        This handles unexpected parameters like 'data' that might come from external systems.
        
        Args:
            level: Log level ("debug", "info", "warning", "error", "critical").
            message: Log message.
            **kwargs: Any additional keyword arguments including 'data'.
        """
        # Extract known parameters
        data = kwargs.get('data', None)
        exc_info = kwargs.get('exc_info', None)
        
        # Format message with data if provided
        log_message = message
        if data:
            log_message = f"{message} {data}"
        
        # Call appropriate logging method
        if level == "debug":
            self._logger.debug(log_message, exc_info=exc_info)
        elif level == "info":
            self._logger.info(log_message, exc_info=exc_info)
        elif level == "warning":
            self._logger.warning(log_message, exc_info=exc_info)
        elif level == "error":
            self._logger.error(log_message, exc_info=exc_info)
        elif level == "critical":
            self._logger.critical(log_message, exc_info=exc_info)
        else:
            self._logger.info(log_message, exc_info=exc_info)