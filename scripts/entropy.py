""" This module calculates the entropy values per topic and plots them.
Functions: 
    get_entropy_values(filename): Get entropy values per topic.
    calculate_entropy(topic_probs): Calculate entropy value.
    average_entropy(doc_topic_dist): Calculate average entropy per topic.
    plot_topic_entropy(topic_entropy_values): Plot entropy values per topic.
"""
from scripts import utils as u
import logging
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy

def get_entropy_values(filename):
    """Get entropy values per topic.
    Args:
        filename: The filename of the dense matrix.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting entropy calculation')

    doc_topic_dist = u.load_dense_matrix(filename)
    topic_entropy_values = average_entropy(doc_topic_dist)
    plot_topic_entropy(topic_entropy_values)
    logger.info('Completed entropy calculation')

def calculate_entropy(topic_probs):
    """ Calculate entropy value.
    Args:
        topic_probs: A list of probabilities.
    Returns:
        entropy_value: A float value.
    """
    # Ensure probabilities sum up to 1 by normalizing
    topic_probs /= np.sum(topic_probs)
    # Remove zeros to avoid log(0) issue
    topic_probs = topic_probs[topic_probs != 0]
    entropy_value = entropy(topic_probs, base=np.e)
    return entropy_value

def average_entropy(doc_topic_dist):
    """ Calculate average entropy per topic.
    Args:
        doc_topic_dist: A dense matrix.
    Returns:
        entropies: A list of entropy values.
    """
    num_topics = doc_topic_dist.shape[1]
    entropies = []
    for i in range(num_topics):
        topic_probs = doc_topic_dist[:, i]
        entropy_value = calculate_entropy(topic_probs)
        entropies.append(entropy_value)
    return entropies

def plot_topic_entropy(topic_entropy_values):
    """ Plot entropy values per topic.
    Args:
        topic_entropy_values: A list of entropy values.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting entropy values per topic')

    topic_label_dict = {0:'protein',
                        1:'vaccine',
                        2:'patient',
                        3:'cell',
                        4:'drug',
                        5:'sample',
                        6:'health'}

    topic_entropy_df = pd.DataFrame({'topic': list(topic_label_dict.values()), 'entropy': topic_entropy_values})

    topic_w_max_entropy = topic_entropy_df.loc[topic_entropy_df['entropy'].idxmax()]
    logger.info('Topic with highest entropy: %s', topic_w_max_entropy)
    topic_w_min_entropy = topic_entropy_df.loc[topic_entropy_df['entropy'].idxmin()]
    logger.info('Topic with lowest entropy: %s', topic_w_min_entropy)

    x = np.arange(len(topic_entropy_values))
    y = np.array(topic_entropy_values)

    plt.scatter(x, y, marker='o', color='black')
    plt.ylim(6, 8)
    plt.yticks(np.arange(6, 8.2, 0.2), ['{:.2f}'.format(i) for i in np.arange(6, 8.2, 0.2)])
    plt.xticks(x, [topic_label_dict[i] for i in x], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlabel('Topic')
    plt.ylabel('Entropy')
    plt.title('Shannon entropy values per topic')

    save_path = 'results/entropy'
    u.create_dir_if_not_exists(save_path)

    plt.savefig(f'{save_path}/entropy_per_topic.png', bbox_inches='tight')
    topic_entropy_df.to_csv(f'{save_path}/entropy_per_topic.csv', index=False)