""" This module contains functions for the ETL process.
Functions:
    collect_ao_metadata: Collect metadata from Semantic Scholar API.
    s2_fetch_metadata: Fetch metadata from Semantic Scholar API.
    create_ao_metadata_df: Create a single dataframe from the files in the directory.
    validate_responses: Validate the responses from the source data and the ao metadata.
    filter_data: Filters and measures the number of common papers between the source data and the ao. metadata instances given the source data paper IDs.
    merge_dfs: Merge the source data and the ao metadata.
    corpus_creation: Corpus formation. This function creates a corpus from a tabular file.
    corpus_preprocessing: Data cleaning. This function lemmatizes the text, removes non-nouns, removes common and uncommon words, and saves the clean corpus.
"""
from scripts import utils as u
import pandas as pd
import requests
import time
import random
random.seed(20191113) # to make the sampling reproducible
import numpy as np
np.set_printoptions(precision=5)
import logging
from tqdm import tqdm
import os
from tmtoolkit.utils import enable_logging
enable_logging()
from tmtoolkit.corpus import (Corpus, save_corpus_to_picklefile, load_corpus_from_picklefile, print_summary, lemmatize, filter_for_pos, to_lowercase,
    remove_punctuation, filter_clean_tokens, remove_common_tokens, remove_uncommon_tokens, tokens_table, dtm)

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

def corpus_creation(filename):
    """ Corpus formation. This function creates a corpus from a tabular file.
    Args:
        filename (str): The filename.
    Returns:
        None 
    """
    logger = logging.getLogger(__name__)
    corpus = Corpus.from_tabular(filename, language='en', id_column='paper_id', text_column='abstract')

    dtm_df, doc_labels, vocab = dtm(corpus, as_table=True, return_doc_labels=True, return_vocab=True)

    logger.info('Raw corpus shape: %s', dtm_df.shape)
    logger.info('Raw corpus vocabulary size: %s', len(vocab))
    logger.info(print_summary(corpus))

    save_path = 'results/corpus'
    u.create_dir_if_not_exists(save_path)
    save_corpus_to_picklefile(corpus, f'{save_path}/raw_corpus.pkl')
    logger.debug('Saved raw corpus to %s', f'{save_path}/raw_corpus.pkl')

def corpus_preprocessing(filename):
    """ Data cleaning. This function lemmatizes the text, removes non-nouns, removes common and uncommon words, and saves the clean corpus.
    Args:
        filename (str): The filename.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    corpus = load_corpus_from_picklefile(filename)
    
    corpus_norm = lemmatize(corpus, inplace=False)
    del corpus
    filter_for_pos(corpus_norm, 'N')
    to_lowercase(corpus_norm)
    remove_punctuation(corpus_norm)
    filter_clean_tokens(corpus_norm, remove_shorter_than=2)
    tokens_df = tokens_table(corpus_norm)

    dtm_df, doc_labels, vocab = dtm(corpus_norm, as_table=True, return_doc_labels=True, return_vocab=True)

    logger.info('Clean corpus shape: %s', dtm_df.shape)
    logger.info('Clean corpus vocabulary size: %s', len(vocab))
    logger.info(print_summary(corpus_norm))

    save_path = 'results/corpus'
    tokens_df.to_csv(f'{save_path}/tokens_table.csv', index=False)

    save_corpus_to_picklefile(corpus_norm, f'{save_path}/clean_corpus.pkl')
    logger.debug('Saved clean corpus to %s', f'{save_path}/clean_corpus.pkl')