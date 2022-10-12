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
    """
    cc = TokenDict()
    random = np.random.default_rng(cfg.random_seed)
    outfile = outdir / "text.txt"

    with open(outfile, "w") as f:
        for path in tqdm(all_csv_in_path(indir)):
            df = pd.read_csv(path, sep=",", low_memory=False)

            # Add row column with unique row tokens
            df = df.reset_index()
            df = df.rename(columns={'index': 'row'})
            df['row'] = path.stem + "_row:" + df['row'].astype(str)

            df = df.applymap(lambda val: int(cc.put(val))).to_numpy()
            for i in range(cfg.num_walks):
                perm = random.permuted(df, axis=1) # Shuffles in-place within rows
                np.savetxt(f, perm, fmt='%i')

    cc.save(outdir / "word_dict")
