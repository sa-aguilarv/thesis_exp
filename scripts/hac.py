"""Perform hierarchical agglomerative clustering.
Funcions:
    get_clusters: Perform hierarchical agglomerative clustering.
    get_max_topic_for_docs: Get the topic number with the highest probability for each document.
    get_umap: Get UMAP visualization.
    get_umap_plot: Create UMAP plot.
    get_topic_cluster_df: Get a DataFrame of the topic clusters.
"""
from scripts import utils as u
from sklearn.cluster import AgglomerativeClustering
import umap
import logging
import random
random.seed(20191113) # to make the sampling reproducible
import numpy as np
np.random.seed(42)
np.set_printoptions(precision=5)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_clusters(filename, params):
    """Perform hierarchical agglomerative clustering.
    Args:
        dense_matrix: A dense matrix.
        params: A dictionary of parameters.
    Returns:
        cluster_labels: A list of cluster labels.
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting hierarchical agglomerative clustering')
    
    doc_topic_dist = u.load_dense_matrix(filename)

    max_topic_for_doc = get_max_topic_for_docs(doc_topic_dist)

    hac = AgglomerativeClustering(n_clusters=params['numClusters'],
                                metric=params['metric'],
                                linkage=params['linkage'])
    y_hac = hac.fit_predict(doc_topic_dist)

    get_topic_cluster_df(max_topic_for_doc, y_hac)

def get_max_topic_for_docs(doc_topic_dist):
    """ Get the topic number with the highest probability for each document.
    Args:
        doc_topic_dist: A dense matrix.
    Returns:
        max_topic_for_doc: A list of topic numbers.
    """
    max_topic_for_doc = np.argmax(doc_topic_dist, axis=1)
    return max_topic_for_doc.tolist()

def get_umap(filename, params):
    """Get UMAP visualization.
    Args:
        filename: The filename of the dense matrix.
        y_hac: The cluster labels.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting UMAP visualization')

    topic_cluster_df = pd.read_csv(filename)
    doc_topic_dist = u.load_dense_matrix('results/tm/7_topics/doc_topic_distr.txt')

    mapper = umap.UMAP(n_neighbors=params['numNeighbors'],
                        random_state=params['randomState'],
                        metric=params['metric'])
    embedding = mapper.fit_transform(doc_topic_dist)

    topic_cluster_df['x'] = embedding[:,0]
    topic_cluster_df['y'] = embedding[:,1]

    topic_cluster_df.to_csv('results/hac/topic_cluster_umap_df.csv', index=False)
    logger.info('Saved topic cluster UMAP DataFrame in results/hac/topic_cluster_umap_df.csv')

def get_umap_plot():
    """ Create UMAP plot.
    Args:
        None
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    filename = 'results/hac/topic_cluster_umap_df.csv'

    scatter_df = pd.read_csv(filename)
    logger.info('Creating UMAP plot')
    logger.info('DF shape: %s', scatter_df.shape)

    plt.figure(figsize=(12,12))
    ax = sns.scatterplot(data=scatter_df,
                                x='x',
                                y='y',
                                hue='cluster',
                                style='topic',
                                palette='Set2',
                                s=20,
                                ec='black')
    
    ax.set(xlabel=None, ylabel=None)
    ax.set_aspect('equal', 'datalim')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([])
    plt.yticks([])
    ax.get_figure().savefig('results/hac/umap.png', bbox_inches='tight')

def get_topic_cluster_df(max_topic_for_doc, y_hac):
    """Get a DataFrame of the topic clusters.
    Args:
        max_topic_for_doc: The topic number with the highest probability for each document.
        y_hac: The cluster labels.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    topic_label_dict = {0:'protein',
                        1:'vaccine',
                        2:'patient',
                        3:'cell',
                        4:'drug',
                        5:'sample',
                        6:'health'}
    
    topic_label_vec = [topic_label_dict[topic] for topic in max_topic_for_doc]

    df = pd.DataFrame({'topic': topic_label_vec, 'cluster': y_hac})
    save_path = 'results/hac'
    u.create_dir_if_not_exists(save_path)
    df.to_csv(f'{save_path}/topic_cluster_df.csv', index=False)
    logger.info('Saved topic cluster DataFrame in %s', save_path)

def describe_clusters():
    logger = logging.getLogger(__name__)
    filename = 'results/hac/topic_cluster_df.csv'
    df = pd.read_csv(filename)
    logger.info('Describing clusters')

    # Create a new column 'topic_count' that contains the count of each topic within its cluster
    df['topic_count'] = df.groupby(['cluster', 'topic'])['topic'].transform('count')

    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        logger.info('Cluster %s: %s', cluster, cluster_df['topic'].value_counts())

    # Plot the distribution of topics in each cluster
    plt.figure(figsize=(12,8))
    sns.countplot(data=df, x='cluster', hue='topic', palette='Set2')
    max_min_tuple = {}
    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        topic_max = cluster_df['topic'].value_counts().idxmax()
        logger.info('Cluster %s: Topic with the most papers: %s', cluster, topic_max)
        topic_min = cluster_df['topic'].value_counts().idxmin()
        logger.info('Cluster %s: Topic with the fewest papers: %s', cluster, topic_min)
        max_min_tuple[cluster] = {'max': topic_max, 'min': topic_min}

    plt.title('Distribution of topics in each cluster', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of papers', fontsize=12)
    plt.legend(title='Topic', title_fontsize=12, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('results/hac/cluster_topic_distribution.png', bbox_inches='tight')

    # save the max and min topic for each cluster in a csv file
    max_min_df = pd.DataFrame(max_min_tuple).T
    max_min_df.to_csv('results/hac/max_min_topic_per_cluster.csv')
    logger.info('Saved max and min topic for each cluster in results/hac/max_min_topic_per_cluster.csv')
    logger.info('Saved cluster topic distribution plot in results/hac/cluster_topic_distribution.png')
