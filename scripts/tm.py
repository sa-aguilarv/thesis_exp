""" Topic modeling functions.
This module contains functions to get the document-term matrix, evaluate topic models, and get topics with LDA model.
Functions:
    get_dtm: Get the document-term matrix.
    get_number_disciplines: Get the number of disciplines.
    get_unique_disciplines: Get the unique disciplines.
    models_evaluation: Evaluate topic models.
    get_topics: Get topics with LDA model.
"""
import logging
import warnings
from scripts import utils as u
import random
random.seed(20191113) # to make the sampling reproducible
import numpy as np
np.set_printoptions(precision=5)
from tqdm import tqdm
import os
from tmtoolkit.utils import enable_logging, disable_logging
from tmtoolkit.corpus import (load_corpus_from_picklefile, dtm)
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.bow.bow_stats import doc_lengths
from tmtoolkit.topicmod.model_stats import generate_topic_labels_from_top_words
import scipy as sp
import pandas as pd
from tmtoolkit.topicmod.visualize import plot_eval_results
import matplotlib.pyplot as plt
from scripts import utils as u

def get_dtm(filename):
    """ Get the document-term matrix.
    Args:
        filename (str): The filename of the corpus.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    corpus = load_corpus_from_picklefile(filename)
    dtm_sparse, doc_labels, vocab = dtm(corpus, return_doc_labels=True, return_vocab=True)

    logger.info('DTM shape: %s', dtm_sparse.shape)

    save_path = 'results/tm'
    u.create_dir_if_not_exists(save_path)
    objects = [doc_labels, vocab]
    for obj, name in zip(objects, ['doc_labels', 'vocab']):
        u.save_object(obj, f'{save_path}/{name}.pkl')

    sp.sparse.save_npz(f'{save_path}/dtm_sparse.npz', dtm_sparse)
    logger.debug('Saved DTM, doc_labels, and vocab in %s', save_path)

def get_number_disciplines():
    """ Get the number of disciplines.
    Args:
        None
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    filename = 'results/data_w_ao_metadata.csv'
    df = pd.read_csv(filename, usecols=['fieldsOfStudy'])
    get_unique_disciplines(df)

def get_unique_disciplines(df):
    """ Get the unique disciplines.
    Args:
        df (pd.DataFrame): The DataFrame.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    fields_ls = df['fieldsOfStudy'].apply(sort_list)

    tota_unique_fields = set()
    for ls in fields_ls:
        unique_fields = set(ls)
        tota_unique_fields.update(unique_fields)
    logger.debug('%s total unique fields of study: %s', len(tota_unique_fields), tota_unique_fields)
    # To get fields frequencies
    # unique_fields= [list(x) for x in set(tuple(x) for x in fields_ls)]
    # logger.debug('%s unique fields of study: %s', len(unique_fields), unique_fields)
    #subject_areas_sizes = df.groupby("fieldsOfStudy").size().reset_index(name="counts")

def sort_list(lst_str):
    """ Sort a list.
    Args:
        lst_str (str): The string representation of a list.
    Returns:
        list: The sorted list.
    """
    lst = eval(lst_str)  # Convert the string representation of list to a list
    sorted_lst = sorted(lst)  # Sort the list
    return sorted_lst

def models_evaluation(params):
    """ Evaluate topic models.
    Args:
        params (dict): The parameters.
    Returns:
        None
    """
    dtm = sp.sparse.load_npz('results/tm/dtm_sparse.npz')

    disable_logging()
    logger = logging.getLogger('lda')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    warnings.filterwarnings('ignore')
    gen_logger = logging.getLogger(__name__)
    
    const_params = {
        'n_iter': params['numIter'],
        'random_state': params['randomState'],
        'eta': params['eta']
    }

    var_params = [{'n_topics': k, 'alpha': 1/k} for k in range(params['minTopics'], params['maxTopics'], params['stepTopics'])]

    metrics = ['loglikelihood', 'coherence_mimno_2011']

    gen_logger.info('Metrics to evaluate: %s', metrics)

    gen_logger.info('Starting evaluation of topic models')
    eval_results = evaluate_topic_models(dtm,
                                        varying_parameters=var_params,
                                        constant_parameters=const_params,
                                        metric=metrics,
                                        return_models=True)

    min_topics = str(params['minTopics'])
    max_topics = str(params['maxTopics'])

    save_path = 'results/tm/eval/topics_' + min_topics + '_' + max_topics
    u.create_dir_if_not_exists(save_path)
    u.save_object(eval_results, f'{save_path}/eval_results.pkl')

    eval_results_by_topics = results_by_parameter(eval_results, 'n_topics')
    u.save_object(eval_results, f'{save_path}/eval_results_by_topics.pkl')

    xaxislabel = 'Number of topics'
    title = ['Perplexity\n(minimize)', 'Topic coherence\n(maximize)']
    yaxislabel = ['perplexity', 'NPMI']

    for index in range(0, len(metrics)):
        fig, subfig, axes = plot_eval_results(eval_results_by_topics,
                                            metric=metrics[index],
                                            xaxislabel=xaxislabel, 
                                            yaxislabel=yaxislabel[index], 
                                            show_metric_direction=False,
                                            figsize=(12, 6),
                                            title=title[index])

        for ax in axes:
            ax.set_title("")
            ax.grid(True)
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.set_xticks(np.arange(params['minTopics'], params['maxTopics'], params['stepTopics']))
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.ticklabel_format(style='sci')
            ax.yaxis.get_offset_text().set_fontsize(16)

        plt.savefig(f'{save_path}/plot_eval_results_{metrics[index]}.png')
        gen_logger.info('Saved %s metric plot', metrics[index])

def get_topics(params):
    """ Get topics with LDA model.
    Args:
        params (dict): The parameters.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    eval_results_by_topics = u.load_object('results/tm/eval/topics_1_20/eval_results_by_topics.pkl')

    best_tm, lda_object = next((item for item in eval_results_by_topics if item[0]['n_topics'] == params['bestK']), None)
    logger.info('Best topic model: %s', best_tm)

    lda_model = lda_object['model']
    doc_topic_distr = lda_model.doc_topic_
    topic_word_distr = lda_model.topic_word_

    save_path = 'results/tm/' + str(params['bestK']) + '_topics'
    u.create_dir_if_not_exists(save_path)

    u.save_dense_matrix(doc_topic_distr, f'{save_path}/doc_topic_distr.txt')
    u.save_dense_matrix(topic_word_distr, f'{save_path}/topic_word_distr.txt')
    logger.info('Saved document-topic and topic-word distributions in %s', save_path)

def get_topic_labels():
    logger = logging.getLogger(__name__)
    vocab_filename = 'results/tm/vocab.pkl'
    dtm_filename = 'results/tm/dtm_sparse.npz'
    topic_word_dist_filename = 'results/tm/7_topics/topic_word_distr.txt'
    doc_topic_dist_filename = 'results/tm/7_topics/doc_topic_distr.txt'

    vocab = u.load_object(vocab_filename)
    dtm= sp.sparse.load_npz(dtm_filename)
    vocab = np.array(vocab)   # we need this to be an array
    doc_len = doc_lengths(dtm)
    topic_word_dist = u.load_dense_matrix(topic_word_dist_filename)
    doc_topic_dist = u.load_dense_matrix(doc_topic_dist_filename)


    topic_labels = generate_topic_labels_from_top_words(
        topic_word_dist,
        doc_topic_dist,
        doc_len,
        np.array(vocab),
        lambda_=0.6
    )

    logger.info('Topic labels: %s', topic_labels)
