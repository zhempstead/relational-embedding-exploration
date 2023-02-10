import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import all_csv_in_path, prev_stage_dir, tee

def intrinsic_downstream(df_x, df_y, outdir, cfg):
    """
    Measure quality of embeddings by intrinsic tasks.
    """
    teefile = outdir / 'results.txt'

    try:
        model_dir = prev_stage_dir(outdir, "graph2model")
    except ValueError:
        model_dir = prev_stage_dir(outdir, "text2model")
    normalize_dir = prev_stage_dir(outdir, "normalize")
    random = np.random.RandomState(cfg.random_seed)

    model_fname = "model"
    if cfg.model_suffix is not None:
        model_fname += f"_{cfg.model_suffix}"
    model = KeyedVectors.load_word2vec_format(model_dir / model_fname)
    word_dict = TokenDict()
    word_dict.load(model_dir / "word_dict.feather")

    results = {'model': [], 'pscore_train': [], 'pscore_test': []}

    with open(teefile, 'w') as fout:
        for csv in all_csv_in_path(normalize_dir):
            tee(fout, f"Evaluating {csv.name}...")
            df = pd.read_csv(csv)
            if len(df.columns) < 2:
                continue

            successes, total = embdi_match_attribute(df, 4, random, model, word_dict)
            results['model'].append(f'{csv.name}_embdi_MA')
            results['pscore_train'].append(total)
            results['pscore_test'].append(successes / total)

            successes, total = embdi_match_row(df, 4, random, model, word_dict)
            results['model'].append(f'{csv.name}_embdi_MR')
            results['pscore_train'].append(total)
            results['pscore_test'].append(successes / total)
            
            for col1 in df.columns:
                tee(fout, f"  Primary column {col1}...")
                for col2 in df.columns:
                    if col1 == col2:
                        continue
                    tee(fout, f"    Secondary column {col2}...")
                    
                    successes, total = embdi_match_concept(df, col1, col2, 4, random, model, word_dict)
                    if total == 0:
                        continue
                    results['model'].append(f'{csv.name}_{col1}_{col2}_embdi_MC')
                    results['pscore_train'].append(total)
                    results['pscore_test'].append(successes / total)

                    successes, total = add_subtract_groups(df, col1, col2, random, model, word_dict)
                    results['model'].append(f'{csv.name}_{col1}_{col2}_as_MC')
                    results['pscore_train'].append(total)
                    results['pscore_test'].append(successes / total)

    return pd.DataFrame(results)


def embdi_match_attribute(df, num_good, seed, model, word_dict):
    successes = 0
    for i in range(20):
        for col in df.columns:
            tokens = list(df[col].sample(n=num_good, random_state=seed))
            token_odd = df.drop(col, axis=1).sample(random_state=seed).sample(axis=1, random_state=seed).iloc[0][0]
            if identifies_odd_one_out(tokens, token_odd, model, word_dict):
                successes += 1
    return successes, len(df.columns)*20

def embdi_match_row(df, num_good, seed, model, word_dict):
    successes = 0
    for i in range(20):
        for row in df.index:
            tokens = list(df.loc[row].sample(n=num_good, random_state=seed))
            token_odd = df[df.index != row].sample(random_state=seed).sample(axis=1, random_state=seed).iloc[0][0]
            if identifies_odd_one_out(tokens, token_odd, model, word_dict):
                successes += 1
    return successes, len(df)*20

def embdi_match_concept(df, col1, col2, num_good, seed, model, word_dict):
    total = 0
    successes = 0
    for token_one, many in df.groupby(col1)[col2]:
        if len(many) < num_good:
            continue
        for i in range(50):
            tokens_many = list(many.sample(n=num_good, random_state=seed))
            token_odd = df.loc[df[col1] != token_one, col2].sample(random_state=seed).iloc[0]
            if identifies_odd_one_out([token_one] + tokens_many, token_odd, model, word_dict):
                successes += 1
            total += 1
    return successes, total

def add_subtract_groups(df, col1, col2, seed, model, word_dict):
    total = 0
    successes = 0
    col1_embs = {}
    col2_avg_embs = {}
    for left, right in df.groupby(col1)[col2]:
        col1_embs[left] = np.expand_dims(model[word_dict.getNumForToken(left)], 0)
        col2_embs = np.array([model[word_dict.getNumForToken(t)] for t in right])
        col2_avg_embs[left] = np.expand_dims(col2_embs.mean(axis=0), 0)
    col1_embs = np.concatenate(list(col1_embs.values()))
    col2_avg_embs = np.concatenate(list(col2_avg_embs.values()))

    col2_avg_minus_embs = col2_avg_embs - col1_embs

    for idx in range(len(col1_embs)):
        shifted = col2_avg_minus_embs + col1_embs[idx]
        similarities = cosine_similarity(col2_avg_embs, shifted)
        predictions = np.argmax(similarities, axis=1)
        # Minus 1 because getting the idx row right doesn't count
        total += len(predictions) - 1
        successes += np.count_nonzero(predictions == idx) - 1
    return successes, total



def identifies_odd_one_out(tokens, token_odd, model, word_dict):
    nums = [word_dict.getNumForToken(t) for t in tokens]
    odd_num = word_dict.getNumForToken(token_odd)
    return model.doesnt_match(nums + [odd_num]) == odd_num


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
