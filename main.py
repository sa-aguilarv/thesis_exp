""" Main script to run the application.
Functions: 
    set_up_logging: Set up the logging configuration.
    main: Main function to run the application.
"""
import os
import logging
import logging.config
from scripts import input_output as io
from scripts import eda
from scripts import utils as u
from scripts import etl
from scripts import tm
import pandas as pd

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
        # Uncomment to list the number of files in each subfolder in ./data
        # files_per_subfolder = eda.get_size()
        # for subfolder, file_count in files_per_subfolder.items():
        #     logger.info(f"Subfolder: {subfolder}, Number of files: {file_count}")
        eda.json_to_df(config['dataPath'])

    elif args.metadata:
        logger.info('Additional metadata collection from S2AG')
        filename = 'results/data.csv'
        results_path = 'results/etl'
        etl.collect_ao_metadata(filename)
        etl.create_ao_metadata_df(results_path)
        etl.filter_data(filename, 'results/ao_metadata.csv')

    elif args.corpus:
        logger.info('Corpus formation and preprocessing')
        filename = 'results/data_w_ao_metadata.csv'
        etl.corpus_creation(filename)
        filename = 'results/corpus/raw_corpus.pkl'
        etl.corpus_preprocessing(filename)

    elif args.tm:
        logger.info('Topic modeling')
        filename = 'results/corpus/clean_corpus.pkl'
        tm.get_dtm(filename)
        
        
if __name__ == '__main__':
    main()