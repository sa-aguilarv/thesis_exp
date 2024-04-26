import logging
from scripts import utils as u
import os

def get_size(dataPath):
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
    # folders = [f for f in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, f))]
    # # get the number of files in each folder
    # size = {}
    # for folder in folders:
    #     logger.debug('Folder: %s', folder)
    #     subfolder = os.listdir(os.path.join(dataPath, folder))
    #     logger.debug('Subfolder: %s', subfolder)
    #     files = os.listdir(os.path.join(dataPath, folder, subfolder))
    #     size[folder] = len(files)
    # logger.info('Size of data: %s', size)
            
        

