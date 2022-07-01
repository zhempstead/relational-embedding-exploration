import argparse
import json
from collections import defaultdict 
from pathlib import Path

import hydra
import numpy as np 
import pandas as pd

from relational_embeddings.utils import all_data_files_in_path


@hydra.main(version_base=None, config_path='../hydra_conf', config_name='preprocess')
def preprocess(cfg):
    '''
    - Make copy of target file with target column removed
    - Remove target column from all other files
    '''
    name = cfg.dataset.name
    target_column = cfg.dataset.target_column
    location = Path(f'./data/{name}')
    base_file = location / 'base.csv'
    processed_file = location / 'base_processed.csv'

    df = pd.read_csv(base_file)
    if target_column in df.columns:
        print(f"Dropping rows from '{base_file}' where '{target_column}' is NA")
        df = df[df[target_column].notna()]
        df.to_csv(base_file, index=False)
        df = df.drop(columns=[target_column])
        print(f"Writing '{processed_file}' without '{target_column}'")
    else:
        print("Writing '{processed_file}'")
    df.to_csv(processed_file, index=False)

    for path in all_data_files_in_path(location):
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)
        if target_column[0] in df.columns: 
            print(f"Dropping column '{target_column}' from '{path}'")
            df = df.drop(columns = target_column)
            df.to_csv(path, index=False)


# This file preprocess the datasets and remove target column from the dataset
def write_textification_strategy(args):
    '''
    Create and write textification strategy for all tables in the dataset
    '''
    with open('./data/data_config.txt', 'r') as json_file:
        configs = json.load(json_file)
        if args.task not in configs: 
            print("No such task")
            return 
   
    config = configs[args.task]
    location = config['location']
    target_file = config['target_file']
    location_processed = config['location_processed']
    target_column = config['target_column']

    # Generate textfication strategies
    strategies = dict()

    for path in all_data_files_in_path(location):
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)

        table_name = path.split("/")[-1]
        strategies[table_name] = get_strategy(df)

    with open("./data/strategies/" + args.task + ".txt", "w") as json_file:
        json.dump(strategies, json_file, indent=4)
    
    dbdir = Path.cwd() / "graph" / args.task
    dbdir.mkdir(parents=True, exist_ok=True)
            

def get_strategy(df):
    '''
    Return strategy dict for given input df
    '''
    strategy = defaultdict(dict)

    for col in df.columns:
        integer_strategy, grain_strategy = "augment", "cell"
        num_distinct_numericals = df[col].nunique()

        if "id" not in col and df[col].dtype in [np.float, np.float16, np.float32, np.float64]:
            if abs(df[col].skew()) >= 2: 
                integer_strategy = "eqw_quantize"
            else:
                integer_strategy = "eqh_quantize"
            
        if df[col].dtype in [np.int64, np.int32, np.int64, np.int]:
            if df[col].max() - df[col].min() >= 5 * df[col].shape[0]:
                if abs(df[col].skew()) >= 2: 
                    integer_strategy = "eqw_quantize"
                else:
                    integer_strategy = "eqh_quantize"

        if df[col].dtype == np.object:
            num_tokens_med = (df[col].str.count(' ') + 1).median()
            if num_tokens_med >= 10: 
                grain_strategy = "token"
            
        strategy[col]["int"] = integer_strategy
        strategy[col]["grain"] = grain_strategy
    


if __name__ == "__main__":
    preprocess()
    #parser = argparse.ArgumentParser() 
    #parser.add_argument('--task', type=str, help='name of the task to be preprocessed')

    #args = parser.parse_args()
    #task_loader(args)
