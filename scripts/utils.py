import logging
import json
import os

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
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        logger = logging.getLogger(__name__)
        logger.info('Created directory: %s', dir)
    else:
        logger = logging.getLogger(__name__)
        logger.info('Directory already exists: %s', dir)

def json_decomposition(data):
    paperId = data['paper_id']
    title = data['metadata']['title']
    abstract = data['abstract'][0]['text']
    back_matter = data['back_matter'][0]['text']
    
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