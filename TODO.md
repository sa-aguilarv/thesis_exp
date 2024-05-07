# Materials and methods

## 1. Database & data collection

- [x] Download data
  - [x] Retrieve dataset using cord-19-tools

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

- [ x] Prepare data
  - [x] Get all applicable abstracts
  - [x] Get metadata --processed json
- [ ] General validation
  - [ ] Describe the data variables/columns
    - [ ] DataFrame shape, size, and data types
    - [ ] Check for missing values
    - [ ] Check for null values
    - [ ] Check for duplicates
    - [ ] Check for outliers
- [ ] Analyze data columns distributions
  - [ ] Authors data
- [ ] Handle missing, null or duplicated values
- [ ] Handle outliers

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
