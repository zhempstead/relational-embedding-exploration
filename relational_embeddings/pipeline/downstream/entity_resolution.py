import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import all_csv_in_path, prev_stage_dir

def entity_resolution_downstream(outdir, cfg):
    """
    Entity resolution.

    Assumes that the dataset consists of
    - An 'er_mapping' csv file not used for training
    - Two additional csv files, with the ID being the first column
    """
    csvs = all_csv_in_path(prev_stage_dir(outdir, "normalize"))
    if len(csvs) != 3:
        raise ValueError("Should be three csv files")

    try:
        model_dir = prev_stage_dir(outdir, "graph2model")
    except ValueError:
        model_dir = prev_stage_dir(outdir, "text2model")
    model_fname = "model"
    if cfg.model_suffix is not None:
        model_fname += f"_{cfg.model_suffix}"
    model = KeyedVectors.load_word2vec_format(model_dir / model_fname)
    word_dict = TokenDict()
    word_dict.load(model_dir / "word_dict.feather")

    df_map = None
    embeddings = []
    id_series = []
    for csv in csvs:
        if csv.name == 'er_mapping.csv':
            df_map = pd.read_csv(csv)
            for col in df_map.columns:
                df_map[col] = df_map[col].map(lambda val: word_dict.getNumForToken(val))
                df_map[col] = df_map[col].astype(int)
        else:
            df = pd.read_csv(csv)
            series = df[df.columns[0]]
            ids = series.map(lambda val: word_dict.getNumForToken(val))
            embs = model[ids]
            ids = ids.astype(int)
            embeddings.append(embs)
            id_series.append(ids)

    emb0, emb1 = embeddings
    id0, id1 = id_series
    similarities = cosine_similarity(emb0, emb1)
    df0 = pd.DataFrame({'rid0': id0})
    df1 = pd.DataFrame({'rid1': id1})
    df0['rid1'] = df1.loc[np.argmax(similarities, axis=1), 'rid1'].reset_index(drop=True)
    df1['rid0'] = df0.loc[np.argmax(similarities, axis=0), 'rid0'].reset_index(drop=True)

    pairs = df0.merge(df1, on=['rid0', 'rid1'])

    true_pairs = pairs.merge(df_map, left_on=list(pairs.columns), right_on=list(df_map.columns))
    true_pairs1 = pairs.merge(df_map, left_on=[pairs.columns[1], pairs.columns[0]], right_on=list(df_map.columns))
    if len(true_pairs1) > len(true_pairs):
        true_pairs = true_pairs1

    true_pos = len(true_pairs)
    false_pos = len(pairs) - true_pos
    false_neg = len(df_map) - true_pos
    true_neg = min(len(df0), len(df1)) - true_pos - false_pos - false_neg

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    if (precision + recall) == 0:
        Fscore = 0
    else:
        Fscore = (2 * precision * recall) / (precision + recall)

    return pd.DataFrame({'model': ["er"], 'pscore_train': [Fscore], 'pscore_test': [Fscore]})

def cosine_similarity(A, B):
    """
    Input: two arrays of shape (rows_A, dim) and (rows_B, dim)

    Output: pairwise cosine similarity of shape (rows_A, rows_B)
    """
    magnitudes = np.outer(rss(A), rss(B))
    return (A @ B.T) / magnitudes


def rss(mat):
    """
    Root sum of squares by row

    Input shape (rows, dim)
    Output shape (rows,)
    """
    return np.sqrt(np.sum(np.square(mat), axis=1))
