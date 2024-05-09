import pandas as pd
import requests
import time
import random
import logging
from tqdm import tqdm
from scripts import utils as u
import os

def collect_ao_metadata(filename):
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
    """
    Query definition to fetch metadata from Semantic Scholar API
    :param base_url: Semantic Scholar API base url [str]
    :param chunk_ids: List of doc_ids to fetch [list]
    :return: List of Semantic Scholar API responses [list of dicts]
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
    # list all files in dir
    logger = logging.getLogger(__name__)
    files = os.listdir(dir)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    logger.debug('Number of files in dir: %s', len(files))

    # read each file and concatenate to a single df
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



def data_cleaning(filename):
    # load paper ids and abstracts only, process abstracts and save clean df to new folder under results
    pass

def calc_error_percentage():
    pass