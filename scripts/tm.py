
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

def get_dtm(filename):
    logger = logging.getLogger(__name__)
    corpus = load_corpus_from_picklefile(filename)
    dtm_df, doc_labels, vocab = dtm(corpus, as_table=True, return_doc_labels=True, return_vocab=True)

    logger.debug('First 10 vocabulary tokens: %s', vocab[:10])
    logger.debug('Document-term matrix shape: %s', dtm_df.shape)

    save_path = 'results/tm'
    u.create_dir_if_not_exists(save_path)
    objects = [doc_labels, vocab]
    for obj, name in zip(objects, ['doc_labels', 'vocab']):
        u.save_object(obj, f'{save_path}/{name}.pkl')
    dtm_df.to_csv(f'{save_path}/dtm.csv', index=False)

    logger.debug('Saved DTM, doc_labels, and vocab in %s', save_path)
    return dtm_df

def estimate_topics(dtm, params):
    disable_logging()
    logger = logging.getLogger('lda')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    warnings.filterwarnings('ignore')
    
    const_params = {
        'n_iter': params['numIter'],
        'random_state': params['randomState'],
        'eta': params['eta']
    }

    var_params = [{'n_topics': k, 'alpha': 1/k} for k in range(params['minTopics'], params['maxTopics'], params['stepTopics'])]

    metrics = ['loglikelihood', 'coherence_mimno_2011']

    eval_results = evaluate_topic_models(dtm,
                                        varying_parameters=var_params,
                                        constant_parameters=const_params,
                                        metric=metrics,
                                        return_models=True)
    logger.debug(eval_results[:3])
