import argparse

def parse_input() -> argparse.Namespace:
    """ Parse input arguments.
    Returns:
        argparse.Namespace: The input arguments.
    """
    parser = argparse.ArgumentParser(description="On Finding Megadiversity Among the Corpus of Scientific Literature")
    parser.add_argument("--eda", action="store_true", help="Exploratory data analysis")
    parser.add_argument("--etl", action="store_true", help="Data processing")
    return parser.parse_args()

def write_output_file(): #TODO: create a function to write the output to a file
    pass