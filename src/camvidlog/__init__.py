import logging

import yaml


def setup_logging(config_filename: str = "logging.yml") -> None:
    # see https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
    with open(config_filename) as config_file:
        config = yaml.safe_load(config_file)
    logging.config.dictConfig(config)
