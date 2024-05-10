""" Exploratory Data Analysis (EDA) module.
This module contains functions to perform EDA on the data.
Functions: 
    get_size: Get the number of files in each subfolder in the dataPath.
    json_to_df: Convert json files to a pandas DataFrame.
    save_df_descriptors: Save the DataFrame descriptors.
"""
import logging
from scripts import utils as u
import os
import pandas as pd
from tqdm import tqdm

def get_size(dataPath = './data'):
    """ Get the number of files in each subfolder in the dataPath.
    Args:
        dataPath (str): The path to the data.
    Returns:
        dict: The number of files in each subfolder.
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading data from %s', dataPath)
    files_per_subfolder = {}
    root_folders = [dataPath]
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                subfolder = os.path.relpath(root, root_folder)
                subfolder_path = os.path.join(root_folder, subfolder)
                files_per_subfolder.setdefault(subfolder_path, 0)
                files_per_subfolder[subfolder_path] += 1
    return files_per_subfolder

def json_to_df(dataPath):
    """ Convert json files to a pandas DataFrame.
    Args:
        dataPath (str): The path to the data.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading data from %s', dataPath)
    files = os.listdir(dataPath)
    logger.debug('Number of files in dataPath: %s', len(files))
    
    new_dir = 'results'
    u.create_dir_if_not_exists(new_dir)
    
    dataList = []
    for file in tqdm(files):
        filename = os.path.join(dataPath, file)
        if file.endswith('.json'):
            try: 
                data = u.open_json_file(filename)
                data = u.json_decomposition(data)
                dataList.append(data)
                    
            except Exception as e:
                #logger.error('Error reading file %s: %s', filename, e)
                continue

    df = pd.DataFrame(dataList)
    logger.info('DF shape: %s', df.shape)
    save_df_descriptors(df)
    df = u.drop_nan_rows(df)
    logger.info('DF shape after dropping rows with NaN values: %s', df.shape)
    df.to_csv(f'{new_dir}/data.csv', index=False)
    logger.info('Data saved to %s', f'{new_dir}/data.csv')
    
def save_df_descriptors(df):
    """ Save the DataFrame descriptors.
    Args:
        df (pd.DataFrame): The DataFrame.
    Returns:
        None
    """
    empty_values = df.isnull().sum()
    save_path = 'results/eda'
    u.create_dir_if_not_exists(save_path)
    empty_values.to_csv(f'{save_path}/empty_values.csv')

