""" This module contains functions to parse input arguments and write output files. 
Functions:
    parse_input: Parse input arguments.
    write_output_file: Write the output to a file.
"""
import argparse

def parse_input() -> argparse.Namespace:
    """ Parse input arguments.
    Returns:
        argparse.Namespace: The input arguments.
    """
    parser = argparse.ArgumentParser(description="On Finding Megadiversity Among the Corpus of Scientific Literature")
    parser.add_argument("--eda", action="store_true", help="Exploratory data analysis")
    parser.add_argument("--metadata", action="store_true", help="Additional metadata collection from S2AG")
    parser.add_argument("--corpus", action="store_true", help="Corpus formation and preprocessing")
    parser.add_argument("--eval", action="store_true", help="Topic modeling")
    parser.add_argument("--lda", action="store_true", help="Get topics with LDA model")
    parser.add_argument("--umap", action="store_true", help="Visualize topics with UMAP")
    return parser.parse_args()

def write_output_file(): #TODO: create a function to write the output to a file
    pass