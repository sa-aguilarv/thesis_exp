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

    # color_dict = {0:"blue", 1:"green", 2:"red", 3:"cyan", 4:"magenta", 5:"yellow", 6:"orange"}
    # cvec = [color_dict[i] for i in topic_cluster_df['cluster']]

    mapper = umap.UMAP(n_neighbors=params['numNeighbors'],
                        random_state=params['randomState'],
                        metric=params['metric'])
    embedding = mapper.fit_transform(doc_topic_dist)

    topic_cluster_df['x'] = embedding[:,0]
    topic_cluster_df['y'] = embedding[:,1]

    topic_cluster_df.to_csv('results/hac/topic_cluster_umap_df.csv', index=False)
    logger.info('Saved topic cluster UMAP DataFrame in results/hac/topic_cluster_umap_df.csv')

    # scatter_dict = {'x': embedding[:,0],
    #                 'y': embedding[:,1],
    #                 'clusters': topic_cluster_df['cluster'],
    #                 'topics': topic_cluster_df['topic']}
    # scatter_df = pd.DataFrame(scatter_dict)

def get_umap_plot():
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

    # Calculate centroids for each cluster, then get the topic corresponding to the centroids
    # n = scatter_df['cluster'].nunique()
    # means = np.vstack([scatter_df[scatter_df['cluster'] == i].mean(axis=0) for i in range(n)])
    # ax = sns.scatterplot(means[:, 0], means[:, 1], hue=range(n), palette='Set2', s=20, ec='black', legend=False, ax=ax)

    # Calculate centroids for each cluster
    #centroids = scatter_df.groupby('topic').mean()

    # Get the most common topic for each cluster
    #centroid_topics = scatter_df.loc[centroids.index, 'topic']
    # centroid_topics = scatter_df.groupby('cluster')['topic'].agg(lambda x: x.value_counts().index[0])
    # Add annotations for centroids with their corresponding topics
    # for cluster, (x, y) in centroids.iterrows():
    #     topic = centroid_topics[cluster]
    #     plt.text(x, y, topic, fontsize=10, ha='center', va='center', color='black')

    # Set the legend
    # handles, labels = sns_plot.get_legend_handles_labels()
    # num_clusters = scatter_df['clusters'].nunique()
    # sns_plot.legend(handles=handles[1:num_clusters+1], labels=labels[1:num_clusters+1])
    # sns_plot.legend(handles=handles[num_clusters+1:], labels=labels[num_clusters+1:])

    # handles, labels = sns_plot.get_legend_handles_labels()
    # unique_clusters = topic_cluster_df['cluster'].unique()
    # unique_topics = topic_cluster_df['topic'].unique()
    # cluster_legend = plt.legend(handles[:len(unique_clusters)],
    #                             [f'Cluster {cluster}' for cluster in unique_clusters],
    #                             title='Clusters',
    #                             loc='upper right')
    # topic_legend = plt.legend(handles[len(unique_clusters):],
    #                         unique_topics, 
    #                         title='Topics', 
    #                         loc='lower right')
    # sns_plot.add_artist(cluster_legend)
    
    #plt.show()
    
    # sns_plot.legend(title='Clusters', loc='upper right')
    # sns_plot.legend(title='Topics', loc='lower right')

    # Handles and labels
    # h, l = sns_plot.get_legend_handles_labels()

    # Legend 1: Clusters
    # leg1 = sns_plot.legend(handles=h[1:8], 
    #                     labels=l[1:8],
    #                     loc='upper right',
    #                     bbox_to_anchor=(1.3, 1))
    # order = [2,1,4,3,5,6,7]
    # leg1 = sns_plot.legend([h[idx] for idx in order],[l[idx] for idx in order],fontsize=14)

    # Legend 2: Topics
    # leg2 = sns_plot.legend(h[9:], l[9:], loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=14)

    #sns_plot.add_artist(leg1)
    # sns_plot.set(xticklabels=[])
    # sns_plot.set_xlabel('')
    # sns_plot.set(xticks=[])
    # sns_plot.set(yticklabels=[])
    # sns_plot.set(yticks=[])
    # sns_plot.set_ylabel('')
    # sns_plot.set_title('')
    
    # plt.tight_layout()
    ax.get_figure().savefig('results/hac/umap.png', bbox_inches='tight')
    


    # plt.scatter(embedding[:, 0], embedding[:, 1], c=cvec, s=0.1, cmap='Spectral')
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