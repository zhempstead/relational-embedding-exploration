from collections import defaultdict
import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from relational_embeddings.lib.utils import all_csv_in_path

DATE_PATTERN = re.compile("(\d+/\d+/\d+)")

def double_join_dataset(outdir, cfg):
    """
    Generate tables consisting of
    - A base table
    - One or more secondary tables
    - One or more tertiary tables per secondary table

    Each secondary table shares a unique foreign key column with the base table (one-to-one), and it
    shares a unique foreign key column with its associated tertiary tables.

    Parameters in cfg:
    - secondary_tables: how many secondary tables
    - string_columns: how many string columns per table
    - unique_tokens: how many unique tokens per row
    - truth_tables: how many secondary tables contain a "truth" column (i.e. a column identical to Y)
    - num_rows: how many rows per table

    The classification task will set the Y column to be identical to the first string column of the
    first tertiarty table corresponding to the first "truth_tables" secondary tables.
    """
    if cfg.secondary_tables <= 0:
        raise ValueError("Must be at least 1 secondary table")
    if cfg.tertiary_tables <= 0:
        raise ValueError("Must be at least 1 tertiary table")
    assert cfg.downstream_task == "classification"
    assert cfg.target_column == "y"
    assert cfg.truth_tables > 0

    random = np.random.default_rng(cfg.random_seed)

    base = pd.DataFrame({"init": range(cfg.num_rows)})
    base = add_string_cols(base, cfg.string_columns, cfg.unique_tokens, "base", random)
    base = base.drop(columns=["init"])
    for fk in range(cfg.secondary_tables):
        base[f"fk_{fk}_id"] = range(cfg.num_rows)

    truth_col = gen_truth_col(cfg.num_rows, cfg.unique_tokens, random)
    base['y'] = "y_" + truth_col
    base.to_csv(outdir / "base.csv", index=False)

    for secondary_id in range(cfg.secondary_tables):
        df = pd.DataFrame({f"fk_{secondary_id}_id": range(cfg.num_rows)})
        df = add_string_cols(df, cfg.string_columns, cfg.unique_tokens, f"secondary_{secondary_id}", random)
        for tertiary_id in range(cfg.tertiary_tables):
            fk_col = f"fk_{secondary_id}_{tertiary_id}_id"
            df[fk_col] = range(cfg.num_rows)

            tdf = pd.DataFrame({fk_col: range(cfg.num_rows)})
            tdf = add_string_cols(tdf, cfg.string_columns, cfg.unique_tokens, f"tertiary_{secondary_id}_{tertiary_id}", random)

            if secondary_id < cfg.truth_tables and tertiary_id == 0:
                tdf["col0"] = f"truth_{secondary_id}_{tertiary_id}_" + truth_col

            tdf.to_csv(outdir / f"tertiary_{secondary_id}_{tertiary_id}.csv", index=False)

        df.to_csv(outdir / f"secondary_{secondary_id}.csv", index=False)

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
