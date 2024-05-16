# On Finding Megadiversity Among the Corpus of Scientific Literature

## Introduction

The objective of this Thesis is to find the most diverse set of scientific papers from a given corpus. The diversity of a set of papers is measured by the number of different topics covered by the set. The set of papers with the highest diversity is called the megadiverse set.

Hence, we address the following problem: given a corpus of scientific papers, find the megadiverse set of papers.

## Requirements

1. OS: Ubuntu LTS 20.04

2. Download [python](https://www.python.org/downloads/) --latest release is fine

3. Install [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)

## Set up work environment

1. Create conda environment with Python 3.8.19

    ```Python
    conda create -n "myenv" python=3.8.19
    ```

2. Activate conda environment

    ```Python
    conda activate myenv
    ```

3. Install requirements

    ```Python
    pip install -r requirements.txt
    ```

4. Download dataset folder from here and add it to the `thesis_exp` project directory.

5. Run application with these parameters in this order:

    ```Python
    python main.py --eda
    ```

    ```Python
    python main.py --metadata
    ```

    ```Python
    python main.py --corpus
    ```

    ```Python
    python main.py --eval
    ```

    ```Python
    python main.py --lda
    ```

    ```Python
    python main.py --umap
    ```

    ```Python
    python main.py --entropy
    ```

    ```Python
    python main.py --biblio
    ```
