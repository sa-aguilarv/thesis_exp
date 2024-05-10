import logging
import json
import pickle
import os
import numpy as np
import pandas as pd

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
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def create_dir_if_not_exists(dir):
    """Create a directory if it does not exist.
    Args:
        dir (str): The directory to create.
    Returns:
        None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def json_decomposition(data):
    """ Decompose the json file into a dictionary.
    Args:
        data (dict): The data from the json file.
    Returns:
        dict: The decomposed data in a dictionary. Keys are paper_id, title, abstract, back_matter, names, affiliations.
    """
    paperId = data['paper_id']
    title = data['metadata']['title']
    
    keys = ['abstract', 'back_matter']
    
    # Validate if paper has abstract and acknowledgements
    check_dict = {}
    for key in keys:
        check = check_key(data, key)
        check_dict[key] = check
    
    meta_dict = {}
    for key in check_dict:
        if check_dict[key] == True:
            meta_dict[key] = data[key][0]['text']
        else:
            meta_dict[key] = None
    
    for key in meta_dict:
        if key == 'abstract':
            abstract = meta_dict[key]
        if key == 'back_matter':
            back_matter = meta_dict[key]

    names = []
    affiliations = []
    for author in data['metadata']['authors']:
        author_name = author['first'] + ' ' + author['last']
        if 'institution' in author['affiliation'] != None:
            author_affiliation = author['affiliation']['institution']
        else:
            author_affiliation = ''
        names.append(author_name)
        affiliations.append(author_affiliation)
        
    affiliations = list(set(affiliations))
    
    data = {'paper_id': paperId, 'title': title, 'abstract': abstract, 'back_matter': back_matter, 'names': names, 'affiliations': affiliations}
    return data

def check_key(dictionary, key):
    """ Check if a key exists in a dictionary.
    Args:
        dictionary (dict): The dictionary.
        key (str): The key.
    Returns:
        bool: True if the key exists, False otherwise.
    """
    value = dictionary.get(key)
    if value is None or value == []:
        return False
    return True

def drop_nan_rows(df):
    """ Drop rows with NaN values.
    Args:
        df (pd.DataFrame): The dataframe.
    Returns:
        pd.DataFrame: The dataframe with NaN rows dropped.
    """
    df = df.replace('None', np.nan)
    df = df.dropna()     
    return df

def after_processing_validation():
    """ Validate the data after processing.
    Returns:
        None
    """
    source_df = pd.read_csv('results/data.csv', usecols=['paper_id'])
    ao_metadata_df = pd.read_csv('results/ao_metadata.csv', usecols=['paperId'])
    ao_metadata_df.rename(columns={'paperId': 'paper_id'}, inplace=True)
    source_w_ao_metadata_df = pd.read_csv('results/data_w_ao_metadata.csv', usecols=['paper_id'])
    cleaned_abstracts_df = pd.read_csv('results/cleaned_abstracts.csv', usecols=['paper_id'])
    names = ['source_df', 'ao_metadata_df', 'source_w_ao_metadata_df', 'cleaned_abstracts_df']

    logger = logging.getLogger(__name__)
    dfList = [source_df, ao_metadata_df, source_w_ao_metadata_df, cleaned_abstracts_df]

    no_common_ids_dict = {}
    for i in range(len(dfList)):
        for j in range(i+1, len(dfList)):
            logger.info(f'Comparing {names[i]} and {names[j]}')
            logger.info(f'Shape of df{i} - df{j}: {dfList[i].shape} - {dfList[j].shape}')
            common_ids = set(dfList[i]['paper_id']).intersection(dfList[j]['paper_id'])
            no_common_ids_dict[(i, j)] = len(common_ids)
    
    for key, value in no_common_ids_dict.items():
        logger.info(f'No. of common paper ids between df{key[0]} and df{key[1]}: {value}')

def save_object(matrix, save_path):
    """ Save an object.
    Args:
        matrix (object): The object to save.
    Returns:
        None
    """
    with open(save_path, 'wb') as f:
        pickle.dump(matrix, f)

def load_object(filename):
    """ Load an object.
    Args:
        filename (str): The filename.
    Returns:
        object: The object.
    """
    with open(filename, 'rb') as f:
        matrix = pickle.load(f)
    return matrix