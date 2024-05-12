# Materials and methods

## 0. Set up work environment

- [x] Create git [repository](https://github.com/sa-aguilarv/thesis_exp)
- [x] Create conda environment: Python 3.8.19
- [x] Set up logging and config file settings

## 1. Database & data collection

- [x] Download data
  - [x] Retrieve dataset using [cord-19-tools](https://pypi.org/project/cord-19-tools/)
  - [x] Complete data download: **25.04.2024**

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
    - [x] Check for duplicates // TODO: Address importance of checking for duplicate paper IDs after sending requests to S2AG
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
    - [x] Complete metadata download: **08.05.2024**
  - [x] Calculate the error percentage
  - [x] Drop rows from data df based on ao metadata df
    - [x] **Validate all non-common paper IDs are removed**
    - [x] **Validate df indexes are reset after merging**

- [x] Data cleaning --deprecated, preferred tmtoolkit methods
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

- [x] Validate data after clean-up process
  - [x] Get no. common IDs in all dfs: source, ao. metadata, source merged with ao. metadata, and cleaned abstracts
  - [x] Get shape for all dfs
  - [x] Validate error percentages

- [x] Data cleaning with tmtoolkit
  - [x] Create Corpus instance
    - Make sure to retrieve the model that was used for NER to clean the corpus
  - [x] Preprocessing
    - [x] Lemmatize
    - [x] Filter for POS (only nouns, 'N')
    - [x] Lowercase
    - [x] Remove tokens shorter than 2
    - [x] Remove common tokens (0.90)
    - [x] Removed uncommon tokens (0.05)
    - [ ] Handle compund words --deprecated, kept only nouns
  - [x] Save raw and clean corpora objects
  - [x] Save tokens table

  ```python
  ERROR - Error: duplicate document label "data_w_ao_metadata-aae1603af1bb84087248441716a5c0bd373603b7" not allowed
  ```

  - [x] Find number of duplicate IDs in the source and ao. metadata dfs
    - This were probably introduced during the ao. metadata download from S2AG
    - [x] Remove duplicate paper IDs from ao. metadata df

### 3.1. Results

- Papers sizes during processing:
  - Source df: 7242
  - Ao. medatata df: 6669
  - Merged df (source and ao. metadata): 4050

- Retrieved the metadata from S2AG in chunks of 500 papers.
  - Ao. metadata df: 6669
  - No. common IDs between source and ao. metadata dfs: 4097
  - Error percentage: **%38.55**

- Filtered ao. metadata df paper IDs information that wasn't in source df:
  - Filtered df: 4050
  - No. common IDs between source and filtered dfs: 4025
  - Error percentage: **%0.61**
  - Error percentage after removing **duplicates (46)**: **%0.0**

- No. publications size after merging data and ao. metadata dfs: 4123 // ~~TODO: Why did the data size increase after merging? There are probably NaN values.~~
  - UPDATE-1: After validation, **I found the number of common IDs between all dfs is 4097**. This suggests that I either (1) didn't drop all uncommon IDs after merging source data with ao. metadata, or (2) didn't reset the indexes after merging.
  - UPDATE-2: The real number of common IDs after merging data and dropping the rows with NaN values is 4025.

- Why does the merged dataframe has a shape of 4050, but the shared IDs are only 4025?
  - UPDATE-1: This is probably related with **parsing errors while handling the papers IDs**. // TODO: Address this in the discussion: This suggests S2AG's paper IDs have issues while being processed with Python. This might be related with why S2AG seems to fail in retrieving some papers metadata by paper IDs.
  - UPDATE-2: **Found out ao. metadata df had duplicate paper IDs**. This suggests that some requests either (1) retrieved the same paper twice, or (2) found two different papers for the same paper ID. // TODO: Address this in the discussion. Further analysis is needed to make sure what happened, but that's out of the scope of this work. Here we will just remove those duplicate paper IDs instances.

### 3.2. Observations

- Even though we are retrieving the metadata from papers using S2 paper IDs, some requests failed and provided the metadata from unrequested paper IDs. Either that or the character matching built-in functions in Python are limited while handling S2 paper IDs format (char).

- The dimension size increased after merging the data and ao. metadata dfs. No NaN values were detected. **The error percentage is probably wrong, re-calculate it later** // ~~TODO: Validate the error percentage function.~~
  - UPDATE-1: The percentage function is correct, the problem related to how the dfs were merged.
  - UPDATE-2: The error percentage function was wrong because it was measuring #source IDs - #common IDs / (#source IDs * 100) instead of **#ao. metadata IDs - #common IDs / (#ao. metadata IDs * 100)**. This since we want to know how many of the IDs in the filtered dataset actually belong to the source dataset.

- The real number of publications after merging source data with ao. metadata dfs and dropping the rows with NaN values was 4025. This means some requests succeeded in finding the matches between the source df paper IDs and the ao. metadata df paper IDs, yet they might retrieved NaN values for one or both of the target columns (year, disciplines).
  - UPDATE-1: It seems that there are two parsing issues regarding S2AG paper IDs:
    1. S2AG fails in some instances to retrieve metadata given a paper ID. Given 7242 paper IDs, it retrieved 6668 instances, and the error percentage was of %38.55. This suggests there is a probability of 0.38 of getting an incorrect response even while using S2AG's defined paper IDs.
    2. Python fails in some instances during merging of dfs based on common S2AG's paper IDs. Given 4050 paper IDs, it successfully matched 4025 instances, and the error percentage was of %0.61. This suggests there is a probability of 0.061 of merging data incorrectly using pandas given common S2AG paper IDs.
  - UPDATE-2: The %0.61 error was related to the duplicate paper IDs. Python was handling duplicate entries because S2AG retrieved them. 

## 4. Topic model

- [x] Get DTM, labels and vocabulary
  - [ ] DTM with CountVectorizer --deprecated, used tmtoolkit instead
  - [x] DTM, labels and vocabulary with tmtoolkit

- [x] Define how to estimate number of topics
  - [x] Analize viability of using [tmtoolkit](https://tmtoolkit.readthedocs.io/en/latest/)
    - [x] Is this the most up-to-date library to accomplish this?
      - I couldn't find any newer libraries, and tmtoolkit has the advantage of having multiple metrics implemented already.
    - [x] Check how they estimated the number of topics in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
      - They considered different numbers of topics: K = 50-2000
      - They fit each model for 1000 iterations since that's when the **log-likelihood values started to converge** // TODO: Identify when the log-likelihood values (see perplexity graph) start to converge with my data
      - They considered three metircs:
        1. Perplexity (minimize)
        2. Topic coherence (maximize)
        3. Topic distinctiveness (maximize)
    - [x] Does tmtoolkit have these metrics? [Yes](https://tmtoolkit.readthedocs.io/en/latest/index.html#topic-modeling)
      - Papers: perplexity ([Griffiths, Steyvers 2004](https://www.pnas.org/doi/full/10.1073/pnas.0307752101)), topic coherence ([Mimno et al. 2011](https://dl.acm.org/doi/10.5555/2145432.2145462)), topic distinctiveness ([Chuang et al. 2012](https://dl.acm.org/doi/10.1145/2254556.2254572)).

- [x] Implement function to estimate the number of topics with tmtoolkit --~~deprecated, tmtoolkit raises errors while running the lda module~~ only works with Python 3.8
  - [x] Install tmtoolkit and needed set up to use chosen evaluation metrics
    - Installed following [guidelines](https://tmtoolkit.readthedocs.io/en/latest/install.html):

    ```Python
    pip install -U tmtoolkit[recommended,lda,topic_modeling_eval_extra] 
    ```

    ```Python
    python -m tmtoolkit setup en
    ```

    - Noticed that tmtoolkit has a more comprehensive [data cleaning pipenline](https://tmtoolkit.readthedocs.io/en/latest/bow.html)
      - **Decided to re-do the data cleaning with tmtoolkit** to check which performs is better.
  - [x] Implement tmtoolkit topic evaluation function with desired metrics

    ```Python
    ModuleNotFoundError: No module named 'lda'
    ```

    - Installed lda package with `pip install lda>=2.0` according to setup file in [repository](https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/setup.py)

    ```Python
    TypeError: cannot convert the series to <class 'int'>
    ```

    - The lda module is still not recognized
    - [x] Decide whether to debug the library or choose another method to apply the topic model
      - Made sure the import was correct, yet the error persisted. Decided to not use tmtoolkit

    - UPDATE-1: Re-installed tmtoolkit packages in a different conda environment

    ```Python
    conda create --name tm-topiceval python=3.8
    ```

    ```Python
    pip install -U tmtoolkit[recommended,lda,sklearn,gensim]
    ```

    ```Python
    python -m tmtoolkit setup en
    ```

    - **SUCCESS*: I was able to run the lda module in tmtoolkit with this separate conda environment.
  
  - [x] [Plot](https://tmtoolkit.readthedocs.io/en/latest/api.html#plot-topic-model-evaluation-results) eval results
    - NOTE: I went to the definition of the `plot_eval_results` function and added preferred formatting: marker and color

    ```Python
      ...
      ax.plot(x, y, label=m, marker='o', color='black')
      ...
    ```

- [ ] Estimate the #topics (K) based on the #disciplines in the corpus, and the loglikehood values. // TODO: Address this in the discussion. We will evaluate this strict disciplines as a social bias. **Are my knowledge claims similar to those in my cluster?**
  - [x] Get the #unique disciplines
    - #unique disciplines: 16
  - [ ] Identify at which #topics the loglikelihood metric converges.
  - [ ] [Save models](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Displaying-and-exporting-topic-modeling-results) and eval results with optimal number of topics

- [ ] Get topics
  - [ ] Run [LDA models](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Evaluation-of-topic-models)
    - [ ] With K = #disciplines
    - [ ] With K = #estimated topics

- [ ] Topics evaluation metrics in tmtoolkit // ~~TODO: Validate viability of using them over scikitlearn's metrics~~
  - [ ] Get [labels for topics](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Generating-labels-for-topics)
    - We use top words to label the topics to get an **abstract representation** of their meaning
  - [ ] Get [marginal topics and word distributions](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Marginal-topic-and-word-distributions)
    - We identify which topics are more important for the corpus, and which words are more important for the topics
    - We get the **topic word distribution** with `best_tm.topic_word_`
  - [ ] Get [word disctinctiveness](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Word-distinctiveness-and-saliency)
    - We identify which words are the most informative words in the corpus
  - [ ] Get [topic word relevance](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Topic-word-relevance)
    - We get the most and least relevant words for each topic considering the corpus
  - [ ] Get [topic coherence](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html#Topic-coherence)
    - We get the coherence of each topic, and identify which are the worst and best topics // TODO: **Which knowledge claims are in these 'worst' and 'best' topics? Which social biases were involved?** Address this in the discussion.

- [ ] Topic aggregation
  - [x] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
    - They aggregated multiple topic models
    - They created clusters using cosine distances. **What is the advantage of using that over Jensen Shannon divergence?** //TODO: Cover this in the discusion
  - [ ] Get cosine distances between papers
  - [ ] Get clusters
    - [ ] Selection of clustering parameters
      - [ ] Check parameters used in that same [paper]([paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533))
    - [ ] Apply Hierarchical agglomerative clustering
  
  - [ ] Projection of inter-article distances
    - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
      - They used UMAP for dimension reduction
    - [ ] Get UMAP plot

### 4.1. Results

- Corpus descriptors:
  - Raw corpus #tokens: 861383
  - Raw corpus vocabulary size: 36773
  - Clean corpus #tokens: 276248
  - Clean corpus vocabulary size: 18920

- Approaches to estimate the #topics:
  - Unique disciplines in dataset: **16**
    - Disciplines: Medicine, Mathematics, Engineering, Sociology, Materials Science, Physics, Environmental Science, Psychology, Computer Science, Chemistry, Business, Economics, Biology, Geography, Political Science, History

### 4.2. Observations

- The corpus vocabulary size (130) and #tokens (99380) were too small after removing common tokens (0.90) and uncommon tokens (0.05). Hence, I decided to not remove these uncommon and common tokens. This resulted in a vocabulary size of 18920

- The distribution of disciplines in the overall dataset is probably skewed. **We didn't balance the dataset since our goal was to find how biases play part in creating interdisciplinary research**. // TODO: Address this in the methodology and discussion. Validate if the distribution is skewed.

## 5. Evaluation of interdisciplinary research

- [ ] Description of publications through scientometrics
  - [ ] Papers yearly distributions
    - [x] Get papers publication year
    - [ ] Get #papers published each year
    - [ ] Plot line diagram: x-year, y-#papers published
  - [ ] Papers disciplines distributions
    - [x] Get papers disciplines
    - [ ] Get #papers per discipline
    - [ ] Create circle plot to visualize disciplines representation in the dataset
  - [ ] Papers country distributions
    - [x] Get papers affiliations
    - [ ] Get countries from papers affiliations
    - [ ] Get #papers per country
    - [ ] Create circle plot to visualize countries representaiton in the dataset
  - [ ] Get authors sex distribution --deprecated, no time

- [ ] Capture the dispersion of disciplines
  - [ ] Get Silhouette values --deprecated, no time
    - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
  - [ ] Get topic entropy values
    - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
    - How diverse are the topics? Which one is the most diverse?
    - Which characteristics do these topics have? e.g., affiliations, disciplines, funding institutions
    - What is the topic entropy over time?

- [ ] Get diagnostic topics --deprecated, no time
  - [ ] Check how they did it in this [paper](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533)
    - They identified which topics are the most characteristic and distinguishing for each discipline, i.e., those that mainly caused the disciplines to become cohesive clusters. They did this based on the Silhouette and topic entropy scores
      - **Are the most diagnostic topics unique to their respective disciplines?** //TODO: Address this in the discussion

- [ ] Get knowledge claims of diagnostic topics --deprecated, no time
  - [ ] Get the corresponding corpus (i.e., set of uncleaned abstracts) of each diagnostic topic
  - [ ] Get the "words that go together" with the 5 top words of each diagnostic topic based on [Point-wise Mutual Information (PMI)](https://tedboy.github.io/nlps/generated/generated/nltk.BigramAssocMeasures.html)

- [ ] Qualitative analysis of knowledge claims --deprecated, no time
  - [ ] For all 5 top words of each diagnostic topic, get the sentences that contain the 5 words with the highest PMI
    - Are these common premises within the field?

## 6. Replicate results

- [ ] Re-run the project and compare original results with these Observations --deprecated, no time

## Additional

- [ ] Review documentation
  - [ ] Verify requirements.txt file

- [ ] Untrack ``TODO.md`` file
