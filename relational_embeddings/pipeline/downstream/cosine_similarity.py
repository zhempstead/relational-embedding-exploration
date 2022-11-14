import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import prev_stage_dir

def cosine_similarity_downstream(df_x, df_y, outdir, cfg):
    """
    Variant of classification task where the Y values appear in embeddings. We simply choose the
    Y token with the closest embedding (by cosine similarity) to the row embedding, and score based
    on whether that was correct or not.
R
    Since this requires the Y tokens to have embeddings it's intended for use in synthetic datasets
    where the Y column is equivalent to one or more "truth" columns.
    """
    teefile = outdir / 'results.txt'
    target_column = df_y.columns[0]

    unique_ys = df_y[target_column].unique()
    # TODO: support multiple truth tables
    y2truth = np.vectorize(lambda y: y.replace('y', 'truth_0'))
    unique_truths = y2truth(unique_ys)

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

    unique_truth_embs = np.array([model[word_dict.getNumForToken(t)] for t in unique_truths])
    x_embs = df_x.to_numpy()
    cs = cosine_similarity(x_embs, unique_truth_embs)
    predictions = unique_ys[np.argmax(cs, axis=1)]
    score = sum(predictions == df_y[target_column]) / len(df_y)
    return pd.DataFrame({'model': ["cosine"], 'pscore_train': [score], 'pscore_test': [score]})


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
