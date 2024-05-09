# Materials and methods

## 0. Set up work environment

- [x] Create git [repository](https://github.com/sa-aguilarv/thesis_exp)
- [x] Create conda environment: Python 3.8.19
- [x] Set up logging and config file settings

## 1. Database & data collection

- [x] Download data
  - [x] Retrieve dataset using [cord-19-tools](https://pypi.org/project/cord-19-tools/)
  - [x] Complete data download: 25.04.2024

    ```Python
    ERROR: OSError: [Errno 28] No space left on device: './custom_license/pdf_json/231960ded0e944a8e87bd5ae4bff74153a1bd113.json' –deprecated
    ```

    - Solution not implemented: `custom_license` dataset won't be used anyway since it contains files with a custom license.

### 1.1. Results

Downloaded datasets:

1. `biorvix_medrxiv`
2. `comm_use_subset`
3. `noncomm_use_subset`
4. `custom_license` --incomplete
5. `pmc_custom_license`

Detailed description:

```Python
INFO - Subfolder: ./data/biorxiv_medrxiv/pdf_json, Number of files: 12
INFO - Subfolder: ./data/comm_use_subset/pdf_json, Number of files: 9918
INFO - Subfolder: ./data/comm_use_subset/pmc_json, Number of files: 9540
INFO - Subfolder: ./data/custom_license/pdf_json, Number of files: 13106
INFO - Subfolder: ./data/noncomm_use_subset/pdf_json, Number of files: 2584
INFO - Subfolder: ./data/noncomm_use_subset/pmc_json, Number of files: 2311
INFO - Subfolder: ./data/pmc_custom_license, Number of files: 1426
TOTAL: 38897
```

### 1.2. Observations

- Not all papers in all datasets are free to use for research purposes.
  1. `biorvix_medrxiv` --no, we want published papers
  2. `comm_use_subset` --only `pdf_json` since `pmc_json` or Pubmed papers lack acknowledgements and authors affiliations.
- “Semantic Scholar updates the dataset every friday, so on fridays and saturdays be sure to redownload data” --taken from [cord-19-tools](https://pypi.org/project/cord-19-tools/)

## 2. Exploratory data analysis (EDA)

- [x] Prepare data
  - [x] Select dataset
    - Chose `/data/comm_use_subset/pdf_json` since `pmc_json` papers lacked affiliations and acknowledgements data
  - [x] Create df based on json metadata: paper ID, title, authors, affiliations, abstract, acknowledgements
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

- [x] Choose social biases to analyze:
  1. geography (affiliations)
  2. PMI words (abstracts)

### 2.1. Results

- Total papers after removing those without abstract nor acknowledgements: 7242

### 2.2. Observations

- All papers from Semantic Scholar that were not PMC papers had affiliations data.
- Regarding handling null values:
  - Papers without abstract: 1163
  - Papers without acknowledgements: 2006
- Total papers dropped because they had NaN values in abstract and/or acknowledgements columns: 2676

## 3. Data processing

- [x] Handle null values for abstract and `back_matter` (i.e., acknowledgements) columns
  - [x] Drop rows with missing abstracts and acknowledgements

- [x] Get missing metadata from [Semantic Scholar S2AG](https://api.semanticscholar.org/api-docs/datasets) given paper ID
  - [x] Create request to S2AG
    - [x] Get year of publication
    - [x] Get the associated disciplines
  - [x] Create ao. metadata df
    - [x] Concatenate ao. metadata df chunks into one df
    - [x] Drop rows with missing year and disciplines

- [x] Validate S2AG responses
  - [x] Get number of common paper IDs between data and ao. metadata df
  - [x] Calculate the error percentage
  - [x] Drop rows from data df based on ao metadata df

- [x] Data cleaning
  - [x] Abstracts: Get meaningful nouns to make up the corpus
    - [x] Lowercase
    - [x] Handle compound words --made them into one word
    - [x] Remove special characters/punctuation, numbers, extra spaces, trailing spaces
    - [x] Remove stopwords
    - [x] Remove non-nouns
    - [x] Lemmatization --no stemming since we won't do sentiment analysis
    - [x] Get unique words
  - [ ] Affiliations: Get list of unique affiliations --deprecated, will do at later stages
  - [ ] Disciplines: Get list of unique disciplines --deprecated, will do at later stages

### 3.1. Results

- Retrieved the metadata from S2AG in chunks of 500 papers. Total: 6669
  - No. common IDs between data and ao. metadata dfs: 4097
  - Error percentage: %43.42
- No. publications size after merging data and ao. metadata dfs: 4123 // NOTE - Why did the data size increase after merging? There are probably NaN values

### 3.2. Observations

- Even though we are retrieving the metadata from papers using S2 paper IDs, some requests failed and provided the metadata from unrequested paper IDs
- The dimension size increased after merging the data and ao. metadata dfs. No NaN values were detected. **The error percentage is probably wrong, re-calculate it later** // NOTE - Validate the error percentage function

## 4. Topic model

- [ ] Data processing
  - [ ] Build TFDF matrix

- [ ] Optimal number of topics
  - [ ] Estimate optimal number of topics with [tmtoolkit](https://tmtoolkit.readthedocs.io/en/latest/)
  - [ ] [Save model](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Displaying-and-exporting-topic-modeling-results) with optimal number of topics

- [ ] Get topics
  - [ ] Apply topic model to the corpus
  - [ ] Save [topic modeling results](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Displaying-and-exporting-topic-modeling-results)

- [ ] Topics evaluation
  - [ ] Get [labels for topics](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Generating-labels-for-topics)
  - [ ] Get [marginal topics and word distributions](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Marginal-topic-and-word-distributions)
  - [ ] Get [word disctinctiveness and saliency](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Word-distinctiveness-and-saliency)
  - [ ] Get [topic word relevance](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Topic-word-relevance)
  - [ ] Get [topic coherence](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Topic-coherence)

- [ ] Topic aggregation
  - [x] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
    - They aggregated multiple topic models
    - They created clusters using cosine distances. **What is the advantage of using that over Jensen Shannon divergence?** //NOTE - Cover this in the discusion
  - [ ] Get cosine distances between papers
  - [ ] Get clusters
    - [ ] Selection of clustering parameters
      - [ ] Check parameters used in that same [paper]([paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533))
    - [ ] Apply Hierarchical agglomerative clustering
  
  - [ ] Projection of inter-article distances
    - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
      - They used UMAP for dimension reduction
    - [ ] Get UMAP plot

## 5. Evaluation of interdisciplinary research

- [ ] Capture the dispersion of disciplines
  - [ ] Get Silhouette values
    - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
  - [ ] Get topic entropy values
    - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)

- [ ] Get diagnostic topics
  - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
    - They identified which topics are the most characteristic and distinguishing for each discipline, i.e., those that mainly caused the disciplines to become cohesive clusters. They did this based on the Silhouette and topic entropy scores
      - **Are the most diagnostic topics unique to their respective disciplines?** //NOTE - Address this in the discussion

- [ ] Get knowledge claims of diagnostic topics
  - [ ] Get the corresponding corpus (i.e., set of uncleaned abstracts) of each diagnostic topic
  - [ ] Get the "words that go together" with the 5 top words of each diagnostic topic based on [Point-wise Mutual Information (PMI)](https://tedboy.github.io/nlps/generated/generated/nltk.BigramAssocMeasures.html)

- [ ] Qualitative analysis of knowledge claims
  - [ ] For all 5 top words of each diagnostic topic, get the sentences that contain the 5 words with the highest PMI
    - Are these common premises within the field?
