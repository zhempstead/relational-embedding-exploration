import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

def all_data_files_in_path(path, include_base=True):
    files = [f for f in path.iterdir()
            if f.is_file() and f.name.endswith('.csv') and not f.name.startswith('base')]
    if include_base:
      files.append(path / 'base_train_x.csv')
    return files

def dataset_dir(dataset_name):
  return Path(__file__).resolve().parent.parent.parent / 'data' / dataset_name
