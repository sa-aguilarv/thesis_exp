import logging
from scripts import utils as u
import os
import pandas as pd
from tqdm import tqdm

def get_size(dataPath = './data'):
    logger = logging.getLogger(__name__)
    logger.info('Loading data from %s', dataPath)
    # list all folders in dataPath
    files_per_subfolder = {}
    root_folders = [dataPath]
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                subfolder = os.path.relpath(root, root_folder)
                subfolder_path = os.path.join(root_folder, subfolder)
                files_per_subfolder.setdefault(subfolder_path, 0)
                files_per_subfolder[subfolder_path] += 1
    return files_per_subfolder

def json_to_df(dataPath):
    logger = logging.getLogger(__name__)
    logger.info('Loading data from %s', dataPath)
    files = os.listdir(dataPath)
    logger.debug('Number of files in dataPath: %s', len(files))
    
    #selected_columns = ['paper_id', 'metadata', 'abstract', 'back_matter']
    #selected_subkeys = {'metadata': 'title', 'abstract': 'text', 'back_matter': 'text'}
    dataList = []
    
    for file in tqdm(files, desc='Loading data', unit='files'):
        filename = os.path.join(dataPath, file)
        if file.endswith('.json'):
            try: 
                data = u.open_json_file(filename)
                paperId = data['paper_id']
                title = data['metadata']['title']
                abstract = data['abstract'][0]['text']
                back_matter = data['back_matter'][0]['text']
                
                names = []
                affiliations = []
                for author in data['metadata']['authors']:
                    author_name = author['first'] + ' ' + author['last']
                    if 'institution' in author['affiliation'] != None:
                        author_affiliation = author['affiliation']['institution']
                    names.append(author_name)
                    affiliations.append(author_affiliation)
                    
                affiliations = list(set(affiliations))
                
                data = {'paper_id': paperId, 'title': title, 'abstract': abstract, 'back_matter': back_matter, 'names': names, 'affiliations': affiliations}
                dataList.append(data)
            except Exception as e:
                logger.error('Error reading file %s: %s', filename, e)
                continue
    
    logger.debug('Data type: %s', type(dataList))
    df = pd.DataFrame.from_records(dataList)
    # create results folder if it does not exist
    new_dir = 'results'
    u.create_dir_if_not_exists(new_dir)
    # Save the dataframe to a csv file
    df.to_csv(f'{new_dir}/data.csv', index=False)
    logger.debug('Dataframe shape: %s', df.shape)
    logger.debug('Dataframe columns: %s', df.columns)
    logger.debug('df[0]: %s', df.iloc[0])
    return df    

