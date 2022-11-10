from collections import defaultdict
import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from relational_embeddings.lib.utils import all_csv_in_path

DATE_PATTERN = re.compile("(\d+/\d+/\d+)")

def single_join_dataset(outdir, cfg):
    """
    Generate tables consisting of
    - A base table
    - One or more secondary tables

    Each secondary table shares a foreign key column with the base table (one-to-one).
    Parameters in cfg:
    - secondary_tables: how many secondary tables
    - string_columns: how many string columns per table
    - unique_tokens: how many unique tokens per row
    - truth_tables: how many secondary tables contain a "truth" column (i.e. a column identical to Y)
    - num_rows: how many rows per table
    - num_fks: how many foreign keys in the base table
      (if 1, all secondary tables have the same foreign key reference.
      If equal to secondary_tables, each secondary table has its own fkey reference to the base table)

    The classification task will set the Y column to be identical to the first string column of the
    first secondary table.
    """
    if cfg.secondary_tables <= 0:
        raise ValueError("Must be at least 1 secondary table")
    if cfg.num_fks > cfg.secondary_tables:
        raise ValueError("Number of foreign keys must be <= the number of secondary tables")
    assert cfg.downstream_task == "classification"
    assert cfg.target_column == "y"
    assert cfg.truth_tables > 0

    base = pd.DataFrame()

    random = np.random.default_rng(cfg.random_seed)

    for fk in range(cfg.num_fks):
        base[f"fk_{fk}_id"] = range(cfg.num_rows)

    truth_col = gen_truth_col(cfg.num_rows, cfg.unique_tokens, random)
    base['y'] = "y_" + truth_col

    base = add_string_cols(base, cfg.string_columns, cfg.unique_tokens, "base", random)
    base.to_csv(outdir / "base.csv", index=False)

    for table_id in range(cfg.secondary_tables):
        fk_id = table_id % cfg.num_fks
        df = pd.DataFrame({f"fk_{fk_id}_id": range(cfg.num_rows)})
        df = add_string_cols(df, cfg.string_columns, cfg.unique_tokens, f"table_{table_id}", random)

        # Overwrite the first column as a truth column (same category pattern as base y)
        if table_id < cfg.truth_tables:
            df["col0"] = f"truth_{table_id}_" + truth_col

        df.to_csv(outdir / f"table_{table_id}.csv")

def gen_truth_col(num_rows, num_unique_tokens, random):
    tiled = np.tile(np.arange(num_unique_tokens), num_rows // num_unique_tokens + 1)[:num_rows]
    return pd.Series(random.permutation(tiled)).astype(str)

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
