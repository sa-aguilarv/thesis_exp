import logging
import json

def load_config(path):
    """Load the config file.
    Args:
        path (str): The path to the config file.
    Returns:
        dict: The config file as a dictionary.
    """
    logger = logging.getLogger(__name__)
    with open(path) as f:
        config = json.load(f)
    logger.debug('Loaded config file: %s', config)
    return config

def open_json_file(filename):
    """ Open a json file.
    Args:
        filename (str): The name of the json file.
    Returns:
        dict: The data from the json file.
    """
    logger = logging.getLogger(__name__)
    with open(filename) as json_file:
        data = json.load(json_file)
    logger.debug('Opened file: %s', filename)
    return data
