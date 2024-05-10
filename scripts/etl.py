""" This module contains functions for the ETL process.
Functions:
    collect_ao_metadata: Collect metadata from Semantic Scholar API.
    s2_fetch_metadata: Fetch metadata from Semantic Scholar API.
    create_ao_metadata_df: Create a single dataframe from the files in the directory.
    validate_responses: Validate the responses from the source data and the ao metadata.
    filter_data: Filters and measures the number of common papers between the source data and the ao. metadata instances given the source data paper IDs.
    data_cleaning: Clean the data.
    handle_abstracts: Handle the abstracts.
    lemmatize: Lemmatize the text.
    remove_non_nouns: Remove non-nouns from the text.
    remove_stopwords: Remove stopwords from the text.
"""
from scripts import utils as u
import pandas as pd
import requests
import time
import random
random.seed(20191113) # to make the sampling reproducible
import numpy as np
np.set_printoptions(precision=5)
import string
import logging
from tqdm import tqdm
import os
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
import spacy
from tmtoolkit.corpus import Corpus
from tmtoolkit.utils import enable_logging
enable_logging()
from tmtoolkit.corpus import (lemmatize, filter_for_pos, to_lowercase,
    remove_punctuation, filter_clean_tokens, remove_common_tokens,
    tokens_table)

def collect_ao_metadata(filename):
    """ Collect metadata from Semantic Scholar API.
    Args:
        filename (str): The filename.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    chunk_size = 500
    base_url = 'https://api.semanticscholar.org/graph/v1/paper/batch'
    
    logger.debug('Reading file: %s', filename)
    df = pd.read_csv(filename, usecols=['paper_id'], index_col=False)
    logger.debug('DF shape: %s', df.shape)
    
    doc_ids = df["paper_id"].tolist()
    chunks = [doc_ids[i:i + chunk_size] for i in range(0, len(doc_ids), chunk_size)]

    for chunk_num, chunk_ids in tqdm(enumerate(chunks, start=1), total=len(chunks)):
        dictList = s2_fetch_metadata(base_url, chunk_ids)
        try:
            cleaned_dictList = [item for item in dictList if item is not None]
            df = pd.DataFrame(cleaned_dictList)
            
            save_path = 'results/etl'
            u.create_dir_if_not_exists(save_path)
            df.to_csv(f'{save_path}/data_{chunk_num}.csv', index=False)
        except Exception as e:
            logger.error('Error reading file %s: %s', filename, e)
            continue

def s2_fetch_metadata(base_url, chunk_ids):
    """ Fetch metadata from Semantic Scholar API.
    Args:
        base_url (str): The base URL.
        chunk_ids (list): The list of IDs.
    Returns:
        dictList (list): The list of dictionaries.
    """
    logger = logging.getLogger(__name__)
    e = 5
    while True:
        try:
            response = requests.post(base_url,
                                    params={'fields': 'year,'
                                            'fieldsOfStudy'},
                                    json={"ids": chunk_ids})
            if response.status_code == 200: # res = str: list of dicts
                dictList = response.json()
                return dictList
            else:
                logger.debug('Status Code: %s', response.status_code)
        except Exception as ex:
            logger.error(f"Error: {ex}")
            if e <= 0:
                break
            e -= 1
            time.sleep(random.randint(60, 65) + (2 ** (5 - e)))

def create_ao_metadata_df(dir):
    """ Create a single dataframe from the files in the directory.
    Args:
        dir (str): The directory.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    files = os.listdir(dir)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    logger.debug('Number of files in dir: %s', len(files))

    dfList = []
    n_rows = 0
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(dir, file))
            df = u.drop_nan_rows(df)
            n_rows += df.shape[0]
            dfList.append(df)
    df = pd.concat(dfList, ignore_index=True)
    logger.info('Total no. papers with ao. metadata: %s', n_rows)
    logger.info('Metadata DF shape: %s', df.shape)
    save_path = 'results/'
    df.to_csv(f'{save_path}/ao_metadata.csv', index=False)

def validate_responses(df, ao_df):
    """ Validate the responses from the source data and the ao metadata.
    Args:
        df (pd.DataFrame): The source data.
        ao_df (pd.DataFrame): The ao metadata.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    df_ids = df['paper_id'].tolist()
    ao_ids = ao_df['paper_id'].tolist()
    common_ids = set(df_ids).intersection(ao_ids)
    logger.debug('Common IDs: %s', len(common_ids))
    # The error percentage measures the number of IDs in the ao metadata that are not in the source data
    error_percentage = (len(ao_ids) - len(common_ids)) / len(ao_ids) * 100
    logger.info('Error percentage: %s', error_percentage)

def filter_data(df_path, ao_df_path):
    """ Filters and measures the number of common papers between the source data and the ao. metadata instances given the source data paper IDs.
    Args:
        df_path (str): The path to the source data.
        ao_df_path (str): The path to the ao metadata.
    Returns:
        None
    """ 
    logger = logging.getLogger(__name__)
    df = pd.read_csv(df_path)
    ao_df = pd.read_csv(ao_df_path)
    logger.debug('Source DF shape: %s', df.shape)
    logger.debug('Ao metadata DF shape: %s', ao_df.shape) 
    
    ao_df.rename(columns={'paperId': 'paper_id'}, inplace=True)
    logger.info('Validation of source and ao metadata')
    validate_responses(df, ao_df)
    
    logger.debug('Counting duplicate paper IDs')
    logger.debug('Source DF duplicates: %s', df.duplicated(subset='paper_id').sum())
    logger.debug('Ao metadata DF duplicates: %s', ao_df.duplicated(subset='paper_id').sum())
    ao_df = ao_df.drop_duplicates(subset='paper_id')

    filtered_df = merge_dfs(df, ao_df)

    filtered_df.to_csv(f'results/data_w_ao_metadata.csv', index=False)

def merge_dfs(df, ao_df):
    """ Merge the source data and the ao metadata.
    Args:
        df (pd.DataFrame): The source data.
        ao_df (pd.DataFrame): The ao metadata.
    Returns:
        pd.DataFrame: The merged dataframe.
    """
    logger = logging.getLogger(__name__)
    filtered_df = pd.merge(df, ao_df, on='paper_id', how='left').reset_index(drop=True)
    filtered_df = u.drop_nan_rows(filtered_df)
    logger.debug('Merged DF shape: %s', filtered_df.shape)
    logger.info('Validation of source and filtered metadata')
    validate_responses(df, filtered_df)
    return filtered_df

def data_cleaning(filename):
    """ Clean the data.
    Args:
        filename (str): The filename.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    try:
        corpus_preprocessing(filename)
        #df_abstracts = handle_abstracts(filename)
        # logger.debug('DF abstracts shape: %s', df_abstracts.shape)
        
        # df_abstracts.to_csv('results/cleaned_abstracts.csv', index=False)
    except Exception as e:
        logger.error('Error: %s', e)

def corpus_preprocessing(filename):
    logger = logging.getLogger(__name__)
    #df = pd.read_csv(filename, usecols=['paper_id', 'abstract'])
    # Create Corpus object from DataFrame
    corpus = Corpus.from_tabular(filename, language='en', id_column='paper_id', text_column='abstract')
    logger.debug('Corpus object: %s', corpus)
    # log corpus 

def handle_abstracts(filename):
    """ Handle the abstracts. This includes cleaning, lemmatization, and removing stopwords.
    Args:
        filename (str): The filename.
    Returns:
        pd.DataFrame: The dataframe with cleaned abstracts.
    """
    logger = logging.getLogger(__name__)
    df = pd.read_csv(filename, usecols=['paper_id', 'abstract'])

    logger.debug('DF shape: %s', df.shape)

    df['abstract'] = df['abstract'].str.lower()
    df['abstract'] = df['abstract'].apply(lambda x: x.replace('-', ''))
    df['abstract'] = df['abstract'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['abstract'] = df['abstract'].str.replace(r'\d+', '') # Removes numbers
    df['abstract'] = df['abstract'].str.replace(r'\n', ' ') # Removes new lines
    df['abstract'] = df['abstract'].str.replace(r'\s+', ' ') # Removes extra spaces
    df['abstract'] = df['abstract'].str.strip() # Removes leading and trailing spaces
    df['abstract'] = df['abstract'].apply(remove_stopwords)
    df['abstract'] = df['abstract'].apply(remove_non_nouns)
    df['abstract'] = df['abstract'].apply(lemmatize)
    df['abstract'] = df['abstract'].apply(lambda x: ' '.join(set(x.split())))
    return df

def lemmatize(text):
    """ Lemmatize the text.
    Args:
        text (str): The text.
    Returns:
        str: The lemmatized text.
    """
    lemmatizer = WordNetLemmatizer()
    token_words = word_tokenize(text)
    lemma_text = ""
    for word in token_words:
        lemma_text = lemma_text + lemmatizer.lemmatize(word) + " "
    return lemma_text

def remove_non_nouns(text):
    """ Remove non-nouns from the text.
    Args:
        text (str): The text.
    Returns:
        str: The text with only nouns. 
    """
    tokenized_text = word_tokenize(text)
    pos_tagged_text = nltk.pos_tag(tokenized_text)
    noun_text = " ".join([word for word, pos in pos_tagged_text if pos in ['NN', 'NNS', 'NNP', 'NNPS']])
    return noun_text

def remove_stopwords(text):
    """ Remove stopwords from the text.
    Args:
        text (str): The text.
    Returns:
        str: The text with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    token_words = word_tokenize(text)
    filtered_text = ""
    for word in token_words:
        if word not in stop_words:
            filtered_text = filtered_text + word + " "
    return filtered_text