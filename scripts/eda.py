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
    
    new_dir = 'results'
    u.create_dir_if_not_exists(new_dir)
    
    dataList = []
    batch_size = 100
    chunks = len(files) // batch_size
    logger.debug('Number of chunks: %s', chunks)
    counter = 0
    for file in tqdm(files):
        filename = os.path.join(dataPath, file)
        if file.endswith('.json'):
            try: 
                data = u.open_json_file(filename)
                data = u.json_decomposition(data)
                dataList.append(data)
                counter += 1
                
                if counter % batch_size == 0:
                    logger.debug('Processed %s files', counter)
                    df.to_csv(f'{new_dir}/data_{chunks}.csv', index=False)
                    logger.debug('Saved chunk %s', chunks)
                    dataList = []
                    counter = 0
                    chunks -= 1
                    
            except Exception as e:
                #logger.error('Error reading file %s: %s', filename, e)
                continue
    
    if dataList:
        df = pd.DataFrame(dataList)
        df.to_csv(f'{new_dir}/data_{chunks}.csv', index=False)

