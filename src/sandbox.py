import os
import json
import logging.config

def setup_logging(default_path='logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
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
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)