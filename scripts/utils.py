import logging
import json
import os
import numpy as np

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