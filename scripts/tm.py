
import logging
import warnings
from scripts import utils as u
import random
random.seed(20191113) # to make the sampling reproducible
import numpy as np
np.set_printoptions(precision=5)
from tqdm import tqdm
import os
from tmtoolkit.utils import disable_logging
from tmtoolkit.corpus import (load_corpus_from_picklefile, dtm)
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.model_stats import (word_distinctiveness, most_distinct_words, least_distinct_words)
import scipy as sp
import pandas as pd

def get_dtm(filename):
    logger = logging.getLogger(__name__)
    corpus = load_corpus_from_picklefile(filename)
    dtm_sparse, doc_labels, vocab = dtm(corpus, return_doc_labels=True, return_vocab=True)

    logger.info('DTM shape: %s', dtm_sparse.shape)

    save_path = 'results/tm'
    u.create_dir_if_not_exists(save_path)
    objects = [doc_labels, vocab]
    for obj, name in zip(objects, ['doc_labels', 'vocab']):
        u.save_object(obj, f'{save_path}/{name}.pkl')

    #dtm_df.to_csv(f'{save_path}/dtm.csv', index=False) #To save as dense matrix
    #np.savetxt(f'{save_path}/dtm.txt', dtm_sparse, delimiter=',', fmt='%1.5f') #To save as dense matrix
    sp.sparse.save_npz(f'{save_path}/dtm_sparse.npz', dtm_sparse)
    logger.debug('Saved DTM, doc_labels, and vocab in %s', save_path)
    return dtm_sparse

def estimate_topics():
    logger = logging.getLogger(__name__)
    filename = 'results/data_w_ao_metadata.csv'
    df = pd.read_csv(filename, usecols=['fieldsOfStudy'])
    get_unique_disciplines(df)

def get_unique_disciplines(df):
    logger = logging.getLogger(__name__)
    fields_ls = df['fieldsOfStudy'].apply(sort_list)

    tota_unique_fields = set()
    for ls in fields_ls:
        unique_fields = set(ls)
        tota_unique_fields.update(unique_fields)
    logger.debug('%s total unique fields of study: %s', len(tota_unique_fields), tota_unique_fields)
        

    # unique_fields= [list(x) for x in set(tuple(x) for x in fields_ls)]
    # logger.debug('%s unique fields of study: %s', len(unique_fields), unique_fields)
    #subject_areas_sizes = df.groupby("fieldsOfStudy").size().reset_index(name="counts")

def sort_list(lst_str):
    """
    Sort the lists
    :param lst_str: String representation of list [str]
    :return: String representation of sorted list [str]
    """
    lst = eval(lst_str)  # Convert the string representation of list to a list
    sorted_lst = sorted(lst)  # Sort the list
    return sorted_lst

# def estimate_topics(dtm, params):
#     disable_logging()
#     logger = logging.getLogger('lda')
#     logger.addHandler(logging.NullHandler())
#     logger.propagate = False
#     warnings.filterwarnings('ignore')
    
#     const_params = {
#         'n_iter': params['numIter'],
#         'random_state': params['randomState'],
#         'eta': params['eta']
#     }

#     var_params = [{'n_topics': k, 'alpha': 1/k} for k in range(params['minTopics'], params['maxTopics'], params['stepTopics'])]

#     metrics = ['loglikelihood', 'coherence_mimno_2011']

#     eval_results = evaluate_topic_models(dtm,
#                                         varying_parameters=var_params,
#                                         constant_parameters=const_params,
#                                         metric=metrics,
#                                         return_models=True)
#     logger.debug(eval_results[:3])
