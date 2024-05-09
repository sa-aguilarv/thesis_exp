from scripts import utils as u
import pandas as pd
import requests
import time
import random
import string
import logging
from tqdm import tqdm
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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

def validate_responses(df, ao_df):
    logger = logging.getLogger(__name__)
    df_ids = df['paper_id'].tolist()
    ao_ids = ao_df['paper_id'].tolist()
    common_ids = set(df_ids).intersection(ao_ids)
    logger.debug('Common IDs: %s', len(common_ids))

    error_percentage = (len(df_ids) - len(common_ids)) / len(df_ids) * 100
    logger.info('Error percentage: %s', error_percentage)


def filter_data(df_path, ao_df_path):
    logger = logging.getLogger(__name__)
    df = pd.read_csv(df_path)
    ao_df = pd.read_csv(ao_df_path)
    logger.debug('DF shape: %s', df.shape)
    logger.debug('AO DF shape: %s', ao_df.shape) 
    
    ao_df.rename(columns={'paperId': 'paper_id'}, inplace=True)
    validate_responses(df, ao_df)

    id_list = ao_df['paper_id'].tolist()
    logger.debug('No. of papers with ao. metadata: %s', len(id_list))

    filtered_df = df[df['paper_id'].isin(id_list)]
    logger.debug('Filtered DF shape: %s', filtered_df.shape)

    filtered_df = pd.merge(filtered_df, ao_df, on='paper_id', how='inner')
    logger.debug('Merged DF shape: %s', filtered_df.shape)

    filtered_df.to_csv(f'results/data_w_ao_metadata.csv', index=False)


def data_cleaning(filename):
    logger = logging.getLogger(__name__)
    try:
        df_abstracts = handle_abstracts(filename)
        logger.debug('DF abstracts shape: %s', df_abstracts.shape)
        df_abstracts.to_csv('results/cleaned_abstracts.csv', index=False)
    except Exception as e:
        logger.error('Error: %s', e)

def handle_abstracts(filename):
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
    lemmatizer = WordNetLemmatizer()
    token_words = word_tokenize(text)
    lemma_text = ""
    for word in token_words:
        lemma_text = lemma_text + lemmatizer.lemmatize(word) + " "
    return lemma_text

def remove_non_nouns(text):
    tokenized_text = word_tokenize(text)
    pos_tagged_text = nltk.pos_tag(tokenized_text)
    noun_text = " ".join([word for word, pos in pos_tagged_text if pos in ['NN', 'NNS', 'NNP', 'NNPS']])
    return noun_text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    token_words = word_tokenize(text)
    filtered_text = ""
    for word in token_words:
        if word not in stop_words:
            filtered_text = filtered_text + word + " "
    return filtered_text