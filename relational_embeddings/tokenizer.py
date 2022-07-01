import json
from collections import defaultdict 
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
import numpy as np 
import pandas as pd

from relational_embeddings.utils import all_data_files_in_path

@hydra.main(version_base=None, config_path='../hydra_conf', config_name='run')
def tokenize(cfg):
    '''
    Convert dataset into tokenized tuples
    '''
    datadir = (Path(get_original_cwd()) / __file__).parent.parent / 'data' / cfg.dataset.name
    # CWD set by hydra
    outdir = Path.cwd() / 'tokenize'
    outdir.mkdir()

    print(f"Tokenizing using '{cfg.tokenization.method}' method...")

    if cfg.tokenization.method == 'leva':
      leva_tokenize(datadir, outdir, cfg)
    else:
      raise ValueError("Unrecognized tokenize method '{cfg.tokenize.method}'")
    
    print(f"Done tokenizing! Output at '{outdir}'")

def leva_tokenize(datadir, outdir, cfg):
    '''
    Tokenization based on Leva paper
    '''
    write_textification_strategy(datadir, outdir)

def write_textification_strategy(datadir, outdir):
    '''
    Create and write textification strategy for all tables in the dataset
    '''
    strategies = dict()
    for path in all_data_files_in_path(datadir):
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)

        strategies[path.name] = get_strategy(df)

    with open(outdir / 'strategy.txt', 'w') as json_file:
        json.dump(strategies, json_file, indent=4)
    
    #dbdir = Path.cwd() / "graph" / args.task
    #dbdir.mkdir(parents=True, exist_ok=True)
            

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
    return strategy


if __name__ == "__main__":
    tokenize()
