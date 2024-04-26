import os
import logging
import logging.config
from scripts import input_output as io
from scripts import eda
from scripts import utils as u

def set_up_logging():
    """Set up the logging configuration.
    Returns:
        logger: A logger object.
    """
    logging.config.fileConfig("logging.ini")
    logger = logging.getLogger(__name__)
    logger.info('Application started in %s', os.getcwd())
    logger.info('Set up logging configuration: level is ' + str(logger.getEffectiveLevel()))
    return logger

def main():
    # Set up the logging configuration
    logger = set_up_logging()
    
    # Load config file
    config = u.load_config('config.json')
    logger.debug('Config file: %s', config)
    
    # Process input and output arguments
    args = io.parse_input()
    logger.debug('Input arguments: %s', args)
    
    if args.eda:
        logger.info('Exploratory data analysis')
        files_per_subfolder = eda.get_size(config['dataPath'])
        for subfolder, file_count in files_per_subfolder.items():
            logger.info(f"Subfolder: {subfolder}, Number of files: {file_count}")
    
if __name__ == '__main__':
    main()