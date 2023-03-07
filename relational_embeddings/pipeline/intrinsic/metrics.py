import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import all_csv_in_path, prev_stage_dir

COLUMN_PAIR_METRIC = {
    'embdi_MA': False,
    'embdi_MR': False,
    'embdi_MC': True,
    'analogy_MA': True,
    'analogy_MR': True,
    'analogy_MC': True,
}

NUM_GOOD=4
REPS=20

ANALOGY_COMPARE_SIZE=100

def metrics_intrinsic(outdir, cfg):
    """
    Measure quality of embeddings by intrinsic tasks.
    """
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

    results = {'metric': [], 'table': [], 'col1': [], 'col2': [], 'successes': [], 'total': []}

    for csv in all_csv_in_path(normalize_dir, exclude_er_map=True):
        print(f"Evaluating {csv.name}...")
        df = pd.read_csv(csv)
        if len(df.columns) < 2:
            print("  Skipping: fewer than 2 columns")
            continue
        embs = np.zeros((len(df), len(df.columns), model.vector_size))
        for idx, col in enumerate(df.columns):
            mapped = df[col].map(lambda val: word_dict.getNumForToken(val))
            na_rows = mapped.isna()
            embs[~na_rows, idx, :] = model[mapped[~na_rows]]
            df.loc[~na_rows, col] = mapped[~na_rows].astype(int)
            # Handle split token rows
            for row in mapped.index[na_rows]:
                try:
                    ids = [word_dict.getNumForToken(val) for val in df.loc[row, col].split()]
                    embs[row, idx, :] = model[ids].mean(axis=0)
                except AttributeError:
                    ids = [-1]
                df.loc[row, col] = int(ids[0])
        ids = df.to_numpy()
        embs = normalize(embs)
        
        for metric, column_pair in COLUMN_PAIR_METRIC.items():
            print(f"  Metric {metric}...")
            mfunc = METRICS[metric]
            if not column_pair:
                successes, total = mfunc(ids, embs, NUM_GOOD, random)
                results['metric'].append(metric)
                results['table'].append(str(csv.name))
                results['col1'].append(None)
                results['col2'].append(None)
                results['successes'].append(successes)
                results['total'].append(total)
                continue

            for idx1, col1 in enumerate(df.columns):
                print(f"    Primary column {col1}...")
                for idx2, col2 in enumerate(df.columns):
                    if idx1 == idx2:
                        continue
                    print(f"      Secondary column {col2}...")
                    successes, total = mfunc(ids, embs, idx1, idx2, NUM_GOOD, random)
                    results['metric'].append(metric)
                    results['table'].append(str(csv.name))
                    results['col1'].append(col1)
                    results['col2'].append(col2)
                    results['successes'].append(successes)
                    results['total'].append(total)


    return pd.DataFrame(results)

def embdi_match_attribute(ids, embs, num_ingroup, seed):
    successes = total = 0
    num_rows, num_cols, dims = embs.shape

    ingroup_indices = seed.randint(0, num_rows, size=num_cols*REPS*NUM_GOOD)
    ingroup_samples = embs[ingroup_indices, REPS*NUM_GOOD*list(range(num_cols)), :].reshape((REPS, NUM_GOOD, num_cols, dims))
    for col in range(num_cols):
        col_idx = np.ones(num_cols, dtype=bool)
        col_idx[col] = False
        outgroup_sample_pool = embs[:, col_idx, :].reshape((num_rows*(num_cols-1), dims))
        outgroup_indices = seed.randint(0, outgroup_sample_pool.shape[0], size=REPS)
        outgroup_samples = outgroup_sample_pool[outgroup_indices, :].reshape((REPS, 1, dims))
        samples = np.concatenate((outgroup_samples, ingroup_samples[:, :, col, :]), axis=1)
        successes += sum(odd_one_out(samples) == 0)
        total += REPS
    return successes, total

def embdi_match_row(ids, embs, num_good, seed):
    successes = total = 0
    num_rows, num_cols, dims = embs.shape

    ingroup_indices = seed.randint(0, num_cols, size=num_rows*REPS*NUM_GOOD)
    ingroup_samples = embs[REPS*NUM_GOOD*list(range(num_rows)), ingroup_indices, :].reshape((REPS, NUM_GOOD, num_rows, dims))
    for row in range(num_rows):
        row_idx = np.ones(num_rows, dtype=bool)
        row_idx[row] = False
        outgroup_sample_pool = embs[row_idx, :, :].reshape(((num_rows-1)*num_cols, dims))
        outgroup_indices = seed.randint(0, outgroup_sample_pool.shape[0], size=REPS)
        outgroup_samples = outgroup_sample_pool[outgroup_indices, :].reshape((REPS, 1, dims))
        samples = np.concatenate((outgroup_samples, ingroup_samples[:, :, row, :]), axis=1)
        successes += sum(odd_one_out(samples) == 0)
        total += REPS
    return successes, total

def embdi_match_concept(ids, embs, col1, col2, num_good, seed):
    successes = total = 0
    num_rows, num_cols, dims = embs.shape

    argsort = ids[:, col1].argsort()
    ids_sorted = ids[:, (col1, col2)][argsort]
    embs_sorted = embs[:, (col1, col2), :][argsort]
    uniq = np.unique(ids_sorted[:, 0], return_index=True)[1]
    if len(uniq) < 2:
        return 0, 0
    col1_embs = embs_sorted[uniq, 0, :]
    col2_emb_groups = np.split(embs_sorted[:, 1, :], uniq[1:])
    for col_id, col1_emb, col2_emb_group in zip(ids_sorted[uniq, 0], col1_embs, col2_emb_groups):
        if col2_emb_group.shape[0] < num_good:
            continue
        group_embs = np.repeat(col1_emb.reshape((dims, 1)), REPS, axis=1).T.reshape((REPS, 1, dims))
        ingroup_indices = seed.randint(0, col2_emb_group.shape[0], size=REPS*NUM_GOOD)
        ingroup_samples = col2_emb_group[ingroup_indices, :].reshape((REPS, NUM_GOOD, dims))
        outgroup_sample_pool = embs_sorted[ids_sorted[:, 0] != col_id, 1, :]
        outgroup_indices = seed.randint(0, outgroup_sample_pool.shape[0], size=REPS)
        outgroup_samples = outgroup_sample_pool[outgroup_indices, :].reshape((REPS, 1, dims))
        samples = np.concatenate((outgroup_samples, ingroup_samples, group_embs), axis=1)
        successes += sum(odd_one_out(samples) == 0)
        total += REPS
    return successes, total


def analogy_match_attribute(ids, embs, col1, col2, num_good, seed):
    successes = total = 0
    num_rows, _, dims = embs.shape

    for row in range(num_rows):
        row2s = seed.randint(0, num_rows, REPS)
        row1_embs = np.repeat(embs[row, col1, :].reshape((dims, 1)), REPS, axis=1).T
        row1_minus_row2s = normalize(row1_embs - embs[row2s, col1, :])
        analogies = row1_minus_row2s + embs[row2s, col2, :]
        correct_embs = np.repeat(embs[row, col2, :].reshape((dims, 1)), REPS, axis=1).T
        correct_similarities = cosine_similarity(analogies, correct_embs)
        all_similarities = pairwise_cosine_similarity(analogies, embs[row, :, :])
        successes += sum(np.max(all_similarities, axis=1) == correct_similarities)
        total += REPS
    print(successes, total)
    return successes, total

def analogy_match_row(ids, embs, col1, col2, num_good, seed):
    successes = total = 0
    num_rows, _, dims = embs.shape

    col1_minus_col2 = normalize(embs[:, col1, :] - embs[:, col2, :])
    for row in range(num_rows):
        row1_embs = np.repeat(col1_minus_col2[row, :].reshape((dims, 1)), REPS, axis=1).T
        row2s = seed.randint(0, num_rows, REPS)
        analogies = row1_embs + embs[row2s, col2, :]
        correct_similarities = cosine_similarity(analogies, embs[row2s, col1, :])
        all_similarities = pairwise_cosine_similarity(analogies, embs[:, col1, :])
        successes += sum(np.max(all_similarities, axis=1) == correct_similarities)
        total += REPS
    
    print(successes, total)
    return successes, total

def analogy_match_concept(ids, embs, col1, col2, num_good, seed):
    successes = total = 0
    num_rows, _, dims = embs.shape
    
    argsort = ids[:, col1].argsort()
    ids_sorted = ids[:, (col1, col2)][argsort]
    embs_sorted = embs[:, (col1, col2), :][argsort]
    uniq = np.unique(ids_sorted[:, 0], return_index=True)[1]
    if len(uniq) < 2:
        return 0, 0
    col1_embs = embs_sorted[uniq, 0, :]
    col2_emb_groups = np.split(embs_sorted[:, 1, :], uniq[1:])
    col2_emb_avgs = [np.mean(group, axis=0).reshape((1, dims)) for group in col2_emb_groups]
    col2_emb_avgs = np.concatenate(col2_emb_avgs, axis=0)

    col2_emb_avgs_minus = col2_emb_avgs - col1_embs
    for idx in range(len(col1_embs)):
        col2_avg_subset = col2_emb_avgs
        col2_minus_subset = col2_emb_avgs_minus
        if len(uniq) > ANALOGY_COMPARE_SIZE:
            rows = list(range(len(uniq)))
            rows.remove(idx)
            sample = seed.choice(rows, ANALOGY_COMPARE_SIZE, replace=False)
            sample = np.concatenate(([idx], sample))
            col2_avg_subset = col2_avg_subset[sample]
            col2_minus_subset = col2_minus_subset[sample]

        shifted = col2_minus_subset + col1_embs[idx]
        similarities = pairwise_cosine_similarity(col2_avg_subset, shifted)
        predictions = np.argmax(similarities, axis=1)
        # Minus 1 because getting the idx row right doesn't count
        total += len(predictions) - 1
        successes += np.count_nonzero(predictions == 0) - 1
    print(successes, total)
    return successes, total


def odd_one_out(A):
    """
    Input: normalized float Array with shape (reps, n, dims)
    Output: Int array with shape (reps)

    For each rep, return the index of the embedding (out of the n samples) that is furthest from the average via cosine similarity
    """
    avg = np.mean(A, axis=1)
    cosines = np.einsum('rnd,rd->rn', A, avg)
    return np.argmin(cosines, axis=1)

def cosine_similarity(A, B):
    """
    Input: two arrays of shape (rows, dim) and (rows, dim)

    Output: cosine similarity of each row, shape (rows,)
    """
    magnitudes = rss(A) * rss(B)
    return np.sum(A*B, axis=1) / magnitudes

def pairwise_cosine_similarity(A, B):
    """
    Input: two arrays of shape (rows_A, dim) and (rows_B, dim)

    Output: pairwise cosine similarity of shape (rows_A, rows_B)
    """
    magnitudes = np.outer(rss(A), rss(B))
    return (A @ B.T) / magnitudes


def rss(A):
    """
    Root sum of squares by row, except 0 becomes 1

    Input shape (rows, dim)
    Output shape (rows,)
    """
    res = np.sqrt(np.sum(np.square(A), axis=1))
    res[np.where(res == 0)] = 1
    return res

def normalize(A, axis=-1):
    """
    Normalize to unit vectors (except zero vectors, which are left alone)
    """
    norm = np.linalg.norm(A, axis=axis, keepdims=True)
    norm[norm == 0] = 1
    return A / norm


METRICS = {
    'embdi_MA': embdi_match_attribute,
    'embdi_MR': embdi_match_row,
    'embdi_MC': embdi_match_concept,
    'analogy_MA': analogy_match_attribute,
    'analogy_MR': analogy_match_row,
    'analogy_MC': analogy_match_concept,
}

