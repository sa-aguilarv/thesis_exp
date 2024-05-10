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
    parser.add_argument("--metadata", action="store_true", help="Collection of papers publication year and discipline from S2AG")
    parser.add_argument("--cleaning", action="store_true", help="Data cleaning")
    parser.add_argument("--tm", action="store_true", help="Topic modeling")
    return parser.parse_args()

def write_output_file(): #TODO: create a function to write the output to a file
    pass