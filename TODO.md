# Materials and methods

## 0. Set up work environment

- [x] Create git repository: https://github.com/sa-aguilarv/thesis_exp
- [x] Create conda environment: Python 3.8.19
- [x] Set up logging and config file settings 

## 1. Database & data collection

- [x] Download data
  - [x] Retrieve dataset using cord-19-tools
  - [x] Complete data download: 25.04.2024

    ```Python
    ERROR: OSError: [Errno 28] No space left on device: './custom_license/pdf_json/231960ded0e944a8e87bd5ae4bff74153a1bd113.json' –deprecated
    ```

    - Solution not implemented: `custom_license` dataset won't be used anyway since it contains files with a custom license.

### Results

Downloaded datasets:

1. biorvix_medrxiv
2. comm_use_subset
3. noncomm_use_subset
4. custom_license
5. pmc_custom_license

Detailed description:

INFO - Subfolder: ./data/biorxiv_medrxiv/pdf_json, Number of files: 12
INFO - Subfolder: ./data/comm_use_subset/pdf_json, Number of files: 9918
INFO - Subfolder: ./data/comm_use_subset/pmc_json, Number of files: 9540
INFO - Subfolder: ./data/custom_license/pdf_json, Number of files: 13106
INFO - Subfolder: ./data/noncomm_use_subset/pdf_json, Number of files: 2584
INFO - Subfolder: ./data/noncomm_use_subset/pmc_json, Number of files: 2311
INFO - Subfolder: ./data/pmc_custom_license, Number of files: 1426
TOTAL: 38897

### Observations

- Not all papers in all datasets are free to use for research purposes.
  - I will use these datasets: biorvix_medrxiv, comm_use_subset/pdf_json, comm_use_subset/pmc_json, i.e., **19470** papers.
- “Semantic Scholar updates the dataset every friday, so on fridays and saturdays be sure to redownload data” --taken from cord-19-tools

## 2. Exploratory data analysis

- [x] Prepare data
  - [x] Select dataset
    - Chose `/data/comm_use_subset/pdf_json` since `pmc_json` papers lacked affiliations and acknowledgements data
  - [x] Create df based on json metadata: paper ID, title, authors, affiliations, abstract, acknowledgements
    - [ ] Get year of publication given paper ID with Semantic Scholar
    - [ ] Transform json to df in batches --deprecated
    - [x] Transform json to df
      - [x] Validate number of papers without abstract or acknowledgements

    ```Python
    scripts.eda - ERROR - 2024-05-07 02:53:57,717 - eda - 975420 - 140251992965504 - Error reading file ./data/comm_use_subset/pdf_json/e642816c09dd07b7bdf515088670a72ee8698bd8.json: list index out of range
    ```

    - Error is related to failed transformations due to missing values.
      - The columns with missing values were 'abstract' and 'back_matter'.

- [x] General validation
  - [x] Describe the data variables/columns
    - [x] DataFrame shape, size, and data types
    - [x] Check for missing values
    - [x] Check for null values
    - [ ] Check for duplicates --deprecated, NA
    - [ ] Check for outliers --deprecated, NA

### Scientometrics

- [x] Choose social biases to analyze: geography (affiliations), PMI words (abstracts)
- [ ] Handle missing, null or duplicated values for 'abstract' and 'back_matter' columns

## 3. Data processing

- [ ] Data pre-processing
  - [ ] Remove stopwords
  - [ ] Handle compound words
  - [ ] Handle nulls
  - [ ] Lemmatization & steeming --deprecated, no time
- [ ] Data processing
  - [ ] Build TFDF matrix

## 4. Topic model

- [ ] Optimal number of topics
- [ ] Topic aggregation
  - [ ] Hierarchical agglomerative clustering
  - [ ] Selection of clustering parameters

## 5. Knowledge network model

- [ ] Distances of discipline centroids
- [ ] Projection of inter-article Distances
- [ ] Cohesiveness and dispersion of disciplines
- [ ] Topic entropy
- [ ] Diagnostic topics

## Evaluation of interdisciplinary research

- [ ] The assessment of interdisciplinarity
- [ ] The assessment of performance
