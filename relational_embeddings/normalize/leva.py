import json
from collections import defaultdict 

import numpy as np 
import pandas as pd

from relational_embeddings.utils import all_data_files_in_path

def leva_normalize(datadir, outdir, cfg):
    '''
    Normalize each table based on leva:
    - Quantize numerical columns
    - Lower case, remove punctuation
    - Either single or multiple tokens per value depending on column grain size
    '''
    strategies = dict()
    for path in all_data_files_in_path(datadir):
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)

        strategy = get_strategy(df)
        strategies[path.name] = strategy

        df = normalize_df(df, strategy, cfg)

        df.to_csv(outdir / path.name, index=False)

    # Write strategies
    with open(outdir / 'strategy.txt', 'w') as json_file:
        json.dump(strategies, json_file, indent=4)

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

def normalize_df(df, strategy, cfg):
    import pdb; pdb.set_trace()
    df = quantize(df, strategy, cfg)
    for col in df.columns:
        lowercase_removepunct(df, col)
        grain = strategy[col]['grain']
        if grain == 'cell':
            df[col] = df[col].str.replace(' ', '_')
    return df

def quantize(df, strategy, cfg):
    num_bins = cfg.num_bins
    # Not enough numerical values for binning
    if df.shape[0] < 2 * num_bins:
        return df 

    bin_percentile = 100.0 / num_bins
    for col in df.columns:
        if df[col].dtype not in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
            continue 
        
        if strategy[col]["int"] == "eqw_quantize":
            bins = [np.percentile(df[col], i * bin_percentile) for i in range(num_bins)]
            df[col] = np.digitize(df[col], bins)
        if strategy[col]["int"] == "eqh_quantize":
            bins = [i * (df[col].max() - df[col].min()) / num_bins for i in range(num_bins)]
            df[col] = np.digitize(df[col], bins)
    return df

def lowercase_removepunct(df, col):
    df[col] = df[col].astype(str)
    df[col] = df[col].str.lower()
    df[col] = df[col].str.replace(',', ' ')
    df[col] = df[col].str.replace('  ', ' ')
    df[col] = df[col].str.strip()


if __name__ == "__main__":
    tokenize()
