import datetime
import logging
import os
import sys
sys.path.append('.')

class MyLogger:
    def __init__(self, log_file=None, log_level=logging.DEBUG, log_format='%(asctime)s %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(log_format)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def clean_logs(self, log_dir):
        now = datetime.datetime.now()
        delta = datetime.timedelta(days=1)
        cutoff_time = now - delta

        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time.timestamp():
                os.remove(filepath)