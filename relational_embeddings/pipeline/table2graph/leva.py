import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import all_csv_in_path


def leva_table2graph(indir, outdir, cfg):
    """
    Build graph leva-style: value/row nodes
    """
    edge_dfs = []
    for path in tqdm(all_csv_in_path(indir, exclude_er_map=True)):
        df = pd.read_csv(path, sep=",", low_memory=False)
        edge_df = make_edge_df(df)
        edge_df["table"] = path.stem
        edge_dfs.append(edge_df)

    edge_df = pd.concat(edge_dfs, ignore_index=True)
    edge_df = add_weights(edge_df)
    edge_df = format_rows(edge_df)

    cc = TokenDict()
    with open(outdir / "edgelist", "w") as edgelist:
        for _, edge in edge_df.iterrows():
            val, row, weight = edge["val"], edge["row"], edge["weight"]
            decoded_val, decoded_row = cc.put(val), cc.put(row)
            edgelist.write(f"{decoded_val} {decoded_row} {weight}\n")
    cc.save(outdir / "node_dict.feather")


def make_edge_df(df):
    """
    Stack df so each row corresponds to one value in the original df, with columns 'row' and 'val'.

    Then, stack further by splitting 'val' into multiple tokens by spaces.
    """
    # Convert into a new DF with a row for each entry in the original DF,
    # with columns 'row', 'col', 'val'
    # stack automatically removes any rows where 'val' is null
    df = df.stack().rename_axis(("row", "col")).reset_index(name="val")
    df.drop("col", axis=1, inplace=True)
    df["row"] = df["row"].astype(str)
    df["val"] = df["val"].astype(str)

    # Split vals into tokens, if applicable
    df["val"] = df["val"].str.split()
    df = df.explode("val", ignore_index=True)

    return df


def add_weights(edge_df):
    """
    - Add weights based on number of value node edges
    """
    links = edge_df.groupby("val")["table"].agg(["size", "nunique"])
    links["weight"] = 1.0 / links["size"]
    links = links.reset_index()
    links = links[["val", "weight"]]

    return edge_df.merge(links, on="val")


def format_rows(edge_df):
    """
    Remove 'table' column but incorporate into row token
    """
    edge_df["row"] = edge_df["table"] + "_row:" + edge_df["row"]
    return edge_df[["row", "val", "weight"]]
