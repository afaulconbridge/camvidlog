import logging.config

import yaml


class ConfigService:
    # hard code config for now
    database_url: str = "sqlite:///app.db"
    framestep: int = 1
    cv_threshold_tracking: float = 0.6
    cv_threshold_detection: float = 0.3
    cv_time_min_tracking: float = 0.5


def setup_logging(config_filename: str = "logging.yml") -> None:
    # see https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)
    logging.config.dictConfig(config)
