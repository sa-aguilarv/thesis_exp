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
np.random.seed(42)
np.set_printoptions(precision=5)
from tqdm import tqdm
import os
from tmtoolkit.utils import enable_logging, disable_logging
from tmtoolkit.corpus import (load_corpus_from_picklefile, dtm)
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.bow.bow_stats import doc_lengths
from tmtoolkit.topicmod.model_stats import generate_topic_labels_from_top_words
from tmtoolkit.topicmod.model_io import ldamodel_top_topic_words
from tmtoolkit.topicmod.visualize import generate_wordclouds_for_topic_words
import scipy as sp
import pandas as pd
from tmtoolkit.topicmod.visualize import plot_eval_results
import matplotlib.pyplot as plt
from scripts import utils as u
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from scipy.cluster.hierarchy import linkage
import plotly.express as px

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
    """ Get topic labels.
    Args:
        None
    Returns:
        None
    """
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

def describe_topics():
    logger = logging.getLogger(__name__)
    logger.info('Calculating topics similarity')
    #get_cosine_sim_between_topics()
    #get_top_100_words()
    get_treemap_for_topics()

def get_treemap_for_topics():
    logger = logging.getLogger(__name__)
    logger.info('Getting treemap for topics')

    filename = 'results/tm/7_topics/topics_top_50_words.csv'
    df = pd.read_csv(filename, index_col=0)
    df = df.T
    df = df.reset_index()
    df = df.rename(columns={'index': 'topic'})
    # drop topic column
    df = df.drop(columns=['topic']) # Each col now represents a topic

    logger.info('Creating treemap for each topic')
    for col in df.columns:
        # split the column name to get the word and its weight
        word_prob_df = df[col].str.split(' ', expand=True)
        word_prob_df.columns = ['Word', 'Probability']
        # remove parentheses from probability column
        word_prob_df['Probability'] = word_prob_df['Probability'].str.replace('(', '')
        word_prob_df['Probability'] = word_prob_df['Probability'].str.replace(')', '')
        word_prob_df['Probability'] = word_prob_df['Probability'].astype(float)
        logger.debug('Word probability DF shape: %s', word_prob_df.shape)

        fig = px.treemap(word_prob_df, 
                        path=['Word'], 
                        values='Probability', 
                        color='Probability',
                        color_continuous_scale='RdBu',
                        title=f'Topic "{col}" top 50 words',
                        color_continuous_midpoint=np.average(word_prob_df['Probability']))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=0, r=0, t=40, b=0))
        fig.write_image(f'results/tm/7_topics/treemap_topic_{col}.png')
        logger.info('Saved treemap for topic %s', col)
        

def get_top_100_words():
    logger = logging.getLogger(__name__)
    logger.info('Getting 10 top words for each topic')
    topic_word_dist_filename = 'results/tm/7_topics/topic_word_distr.txt'
    topic_word_dist = u.load_dense_matrix(topic_word_dist_filename)
    
    logger.debug('Topic-word distribution shape: %s', topic_word_dist.shape)

    # Create a df for the first 10 words of each topic
    vocab = u.load_object('results/tm/vocab.pkl')

    topic_label_dict = {0:'protein',
                    1:'vaccine',
                    2:'patient',
                    3:'cell',
                    4:'drug',
                    5:'sample',
                    6:'health'}

    df = ldamodel_top_topic_words(topic_word_distrib=topic_word_dist,
                                vocab=vocab,
                                top_n=50)
    df.columns = [np.arange(1, len(df.columns) + 1)]
    df.index = [topic_label_dict[i] for i in topic_label_dict]
    df.to_csv(f'results/tm/7_topics/topics_top_50_words.csv', index=True)

    

def get_cosine_sim_between_topics():
    """ Get the cosine similarity matrix from the document-topic distribution, and plot a heatmap.
    Args:
        None
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    filename = 'results/tm/7_topics/doc_topic_distr.txt'
    doc_topic_dist = u.load_dense_matrix(filename)

    logger.debug('Document-topic distribution shape: %s', doc_topic_dist.shape)

    similarity_matrix = cosine_similarity(doc_topic_dist.T)

    # Set the diagonal of the similarity matrix to 0
    np.fill_diagonal(similarity_matrix, 0)

    logger.debug('Cosine similarity matrix shape: %s', similarity_matrix.shape)

    plt.figure(figsize=(10, 10))
    topic_label_dict = {0:'protein',
                        1:'vaccine',
                        2:'patient',
                        3:'cell',
                        4:'drug',
                        5:'sample',
                        6:'health'}
    
    # Create a clustermap with the row dendrogram
    row_linkage = linkage(similarity_matrix, method='average')

    cmap = sns.light_palette("blue", reverse=True, as_cmap=True)
    sns.set_theme(font_scale=1.4)
    g = sns.clustermap(similarity_matrix,
                        row_linkage=row_linkage,
                        cmap=cmap,
                        vmin=0, 
                        vmax=0.18, 
                        xticklabels=[topic_label_dict[i] for i in range(7)],
                        yticklabels=[topic_label_dict[i] for i in range(7)], annot=True, 
                        fmt=".2f", 
                        annot_kws={"size": 16},
                        cbar_kws={'label': 'cosine distance', 'ticks': [0.04, 0.08, 0.12, 0.16]})

    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # Rotate y-tick labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)  # Rotate x-tick labels
    plt.savefig('results/tm/7_topics/cosine_similarity.png', bbox_inches='tight')
