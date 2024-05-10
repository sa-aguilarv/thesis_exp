
import logging
from scripts import utils as u
import random
random.seed(20191113) # to make the sampling reproducible
import numpy as np
np.set_printoptions(precision=5)
from tqdm import tqdm
import os
from tmtoolkit.utils import enable_logging
from tmtoolkit.corpus import (Corpus, save_corpus_to_picklefile, load_corpus_from_picklefile, print_summary, dtm)

def get_dtm(filename):
    logger = logging.getLogger(__name__)
    corpus = load_corpus_from_picklefile(filename)
    dtm_df, doc_labels, vocab = dtm(corpus, as_table=True, return_doc_labels=True, return_vocab=True)

    logger.debug('First 10 document labels: %s', doc_labels[:10])
    logger.debug('First 10 vocabulary tokens: %s', vocab[:10])
    logger.debug('Document-term matrix shape: %s', dtm_df.shape)

    save_path = 'results/tm'
    u.create_dir_if_not_exists(save_path)
    objects = [doc_labels, vocab]
    for obj, name in zip(objects, ['doc_labels', 'vocab']):
        u.save_object(obj, f'{save_path}/{name}.pkl')
    dtm_df.to_csv(f'{save_path}/dtm.csv', index=False)

    logger.debug('Saved DTM, doc_labels, and vocab in %s', save_path)
