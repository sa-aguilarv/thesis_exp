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
    # Get the topic number with the highest probability for each document
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

    color_dict = {0:"blue", 1:"green", 2:"red", 3:"cyan", 4:"magenta", 5:"yellow", 6:"orange"}
    cvec = [color_dict[i] for i in topic_cluster_df['cluster']]

    mapper = umap.UMAP(n_neighbors=params['numNeighbors'],
                        random_state=params['randomState'],
                        metric=params['metric'])
    embedding = mapper.fit_transform(doc_topic_dist)

    scatter_dict = {'x': embedding[:,0],
                    'y': embedding[:,1],
                    'clusters': cvec,
                    'topics': topic_cluster_df['topic']}
    scatter_df = pd.DataFrame(scatter_dict)

    sns_plot = sns.scatterplot(data=scatter_df,
                                x='x',
                                y='y',
                                hue='clusters',
                                style='topics')
    
    handles, labels = sns_plot.get_legend_handles_labels()

    #leg = sns_plot.legend(handles=handles[1:], 
                        # labels=labels[1:],
                        # loc='upper right',
                        # bbox_to_anchor=(1.3, 1))

    #sns_plot.add_artist(leg)
    sns_plot.set(xticklabels=[])
    sns_plot.set_xlabel('')
    sns_plot.set(xticks=[])
    sns_plot.set(yticklabels=[])
    sns_plot.set(yticks=[])
    sns_plot.set_ylabel('')
    sns_plot.set_title('')

    


    # plt.scatter(embedding[:, 0], embedding[:, 1], c=cvec, s=0.1, cmap='Spectral')
    plt.show()

    # plt.scatter(embedding[:, 0], standard_embedding[:, 1], c=mnist.target.astype(int), s=0.1, cmap='Spectral')
    # u.plot_umap(embedding, y_hac)

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