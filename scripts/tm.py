from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import logging
from scripts import utils as u

def get_dtm(df):
    logger = logging.getLogger(__name__)
    cv = CountVectorizer()
    data_cv = cv.fit_transform(df['abstract'])

    save_path = 'results/tm'
    u.create_dir_if_not_exists(save_path)
    u.save_object(data_cv, save_path + '/dtm.pkl')
    logger.info('Data-term matrix saved in %s', f'{save_path}/dtm.pkl')
    return data_cv

def get_topics(filename):
    logger = logging.getLogger(__name__)
    df = pd.read_csv(filename, usecols=['abstract'])
    dtm = get_dtm(df)
    logger.debug('Document-term matrix shape is %s', dtm.shape)

    # LDA


    
    

    

def get_optimal_number_of_topics():
    pass

