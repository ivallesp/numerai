__author__ = "ivallesp"

import os
import json
import logging.config
import sys
from src.common_paths import get_logs_path

def setup_logging_environment(default_path='logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    :default_path: path to the configuration file (str|unicode)
    :default_level: log level to be recorded (logging.SOMETHING)
    :env_key: environment variable containing the path of the logging.json config file. If default_path is specified,
    it takes no effect.
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    with open(path, 'rt') as f:
        config = json.load(f)
    info_filename = config["handlers"]["info_file_handler"]["filename"]
    config["handlers"]["info_file_handler"]["filename"] = os.path.join(get_logs_path(), info_filename)
    error_filename = config["handlers"]["error_file_handler"]["filename"]
    config["handlers"]["error_file_handler"]["filename"] = os.path.join(get_logs_path(), error_filename)
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    logger.info("Logger initialized! Storing them in path {}".format(get_logs_path()))


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Function used to dump write the exceptions into the error log.
    """
    logger = logging.getLogger("excepthook")
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

