import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import all_csv_in_path


def naive_table2text(indir, outdir, cfg):
    """
    Create text from randomized row values (with row nodes)

    Exactly one of the following params must not be None:
    - num_walks: the number of sentences created from each row
    - token_repeats: a value for num_walks will be chosen such that the average repeats for each
                     token will be approximately equal to this value. If you set this to N*L then
                     the output text corpus will be approximately the same size as that of a Leva
                     random walk with walk length L and num_walks N.
    """
    if (cfg.num_walks is None) == (cfg.token_repeats is None):
        raise ValueError("Specify exactly one of num_walks and token_repeats in the config")
    cc = TokenDict()
    random = np.random.default_rng(cfg.random_seed)
    outfile = outdir / "text.txt"

    preloaded_token_dict = False

    num_walks = cfg.num_walks
    if num_walks is None:
        print("Checking corpus size...")
        single_pass_size = 0
        for path in tqdm(all_csv_in_path(indir)):
            df = pd.read_csv(path, sep=",", low_memory=False)

            if cfg.add_row_nodes:
                df = df.reset_index()
                df = df.rename(columns={'index': 'row'})
                df['row'] = path.stem + "_row:" + df['row'].astype(str)
        
            single_pass_size += len(df) * len(df.columns)
            df = df.applymap(lambda val: int(cc.put(val))).to_numpy()

        corpus_size_goal = len(cc.token2id) * cfg.token_repeats
        num_walks = max(1, round(corpus_size_goal / single_pass_size))
        print(f"Repeat goal: {cfg.token_repeats}. Unique tokens: {len(cc.token2id)}. Num walks: {num_walks}. Corpus size: {single_pass_size * num_walks}")
        preloaded_token_dict = True

    with open(outfile, "w") as f:
        if preloaded_token_dict:
            dictput = lambda val: int(cc.getNumForToken(val))
        else:
            dictput = lambda val: int(cc.put(val))

        for path in tqdm(all_csv_in_path(indir)):
            df = pd.read_csv(path, sep=",", low_memory=False)

            # Add row column with unique row tokens
            if cfg.add_row_nodes:
                df = df.reset_index()
                df = df.rename(columns={'index': 'row'})
                df['row'] = path.stem + "_row:" + df['row'].astype(str)

            df = df.applymap(dictput).to_numpy()
            for i in range(num_walks):
                perm = random.permuted(df, axis=1) # Shuffles in-place within rows
                np.savetxt(f, perm, fmt='%i')

    cc.save(outdir / "word_dict.feather")
