import os
from os import listdir
from os.path import isfile, join


def all_data_files_in_path(path):
    return [f for f in path.iterdir()
            if f.is_file() and f.name.endswith('.csv') and f.name != 'base.csv']
