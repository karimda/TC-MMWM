"""
Logging Utilities for TC-MMWM
-----------------------------
Provides structured logging for training, evaluation, and deployment.
Supports console output, file logging, and integration with tensorboard.
"""

import os
import logging
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", log_file=None, level=logging.INFO):
        """
        Initializes a logger for TC-MMWM experiments.

        Args:
            log_dir (str): Directory to store log files
            log_file (str): Specific log filename (optional)
            level (int): Logging level (default: INFO)
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_file is None:
            log_file = f"tc_mmwm_{timestamp}.log"

        self.log_path = os.path.join(log_dir, log_file)

        # Configure logger
        self.logger = logging.getLogger("TC-MMWM")
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(ch_formatter)

        # File handler
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(level)
        fh_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        fh.setFormatter(fh_formatter)

        # Add handlers
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def info(self, message):
        """Logs an informational message."""
        self.logger.info(message)

    def warning(self, message):
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Logs an error message."""
        self.logger.error(message)

    def debug(self, message):
        """Logs a debug message."""
        self.logger.debug(message)

    def get_log_path(self):
        """Returns the path to the log file."""
        return self.log_path


# Example usage
if __name__ == "__main__":
    log = Logger(log_dir="logs", log_file="example.log")
    log.info("Starting TC-MMWM experiment")
    log.debug("Debugging latent state dimensions")
    log.warning("This is a test warning")
    log.error("This is a test error")
