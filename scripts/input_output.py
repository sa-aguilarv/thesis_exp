import argparse

def parse_input() -> argparse.Namespace:
    """ Parse input arguments.
    Returns:
        argparse.Namespace: The input arguments.
    """
    parser = argparse.ArgumentParser(description="On Finding Megadiversity Among the Corpus of Scientific Literature")
    parser.add_argument("--eda", action="store_true", help="Exploratory data analysis")
    parser.add_argument("--etl", choices=['train', 'test'], help="Extract, transform, and load data")
    parser.add_argument("--sampling", action="store_true", help="Sampling data")
    parser.add_argument("--topic_modeling", action="store_true", help="Topic modeling")
    parser.add_argument("--topic_aggregation", action="store_true", help="Topic aggregation")
    parser.add_argument("--adjacency_matrix", action="store_true", help="Create adjacency matrix")
    parser.add_argument("--network_analysis", action="store_true", help="Network analysis")
    return parser.parse_args()

def write_output_file(): #TODO: create a function to write the output to a file
    pass