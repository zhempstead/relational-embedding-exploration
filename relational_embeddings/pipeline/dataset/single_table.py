from collections import defaultdict
import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from relational_embeddings.lib.utils import all_csv_in_path

def single_table_dataset(outdir, cfg):
    """
    Generate a base table consisting of a variable number of string columns, and a Y column.
    Optionally, there can be secondary tables but no tables are connected by foreign keys.
    
    Each secondary table shares a foreign key column with the base table (one-to-one).
    Parameters in cfg:
    - secondary_tables: how many secondary tables (can be 0)
    - string_columns: how many string columns per table
    - unique_tokens: how many unique tokens per row
    - truth_columns: how many string columns in the base table determine the Y values
    - num_rows: how many rows per table

    The regression task will set the Y column to be equal to the sum of the numerical portions of
    the values in the truth columns.
    """
    assert cfg.downstream_task == "regression"
    assert cfg.target_column == "y"
    assert cfg.truth_columns <= cfg.string_columns
    assert cfg.truth_columns > 0

    random = np.random.default_rng(cfg.random_seed)
    
    base = pd.DataFrame()
    base["init"] = range(cfg.num_rows)
    base = add_string_cols(base, cfg.string_columns, cfg.unique_tokens, "base", random)
    base = base.drop(columns=["init"])

    base = add_ycol(base, cfg.truth_columns)
    base.to_csv(outdir / "base.csv", index=False)

    for table_id in range(cfg.secondary_tables):
        df = pd.DataFrame({"init": range(cfg.num_rows)})
        df = add_string_cols(df, cfg.string_columns, cfg.unique_tokens, f"table_{table_id}", random)
        df = df.drop(columns=['init'])

        df.to_csv(outdir / f"table_{table_id}.csv", index=False)

def add_ycol(df, num_truth_columns):
    df['y'] = 0.0
    for colid in range(num_truth_columns):
        col = df.columns[colid]
        df['y'] += df[col].str.split('_').str[-1].astype(int)
    return df

def add_string_cols(df, num_cols, num_unique_tokens, prefix, random):
    '''
    Add the specified number of string columns to df and return it
    '''
    num_rows = len(df)
    tiled = np.tile(np.arange(num_unique_tokens), num_rows // num_unique_tokens + 1)[:num_rows]
    for cid in range(num_cols):
        cname = f"col{cid}"
        tprefix = f"{prefix}_{cname}_"
        ids = pd.Series(random.permutation(tiled))
        df[cname] = tprefix + ids.astype(str)
    return df
