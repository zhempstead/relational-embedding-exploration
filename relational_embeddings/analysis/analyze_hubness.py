import itertools
from pathlib import Path
import sys

import pandas as pd

GROUP_COLS = ['text2model.dimensions', 'model2emb.use_value_nodes']

def main(experiment_dir):
    results = experiment_dir / 'results.csv'
    row_hubness = experiment_dir / 'hubness_rows.csv'
    full_hubness = experiment_dir / 'hubness_all.csv'
    outfile = experiment_dir / 'hubness_analysis.csv'

    df = pd.read_csv(results)
    rh = pd.read_csv(row_hubness)
    fh = pd.read_csv(full_hubness)

    dfr = df.merge(rh)
    dff = df.merge(fh)

    group_cols = [c for c in GROUP_COLS if c in df.columns]

    print(f"Row hubness correlations (by {group_cols}):")
    corr_dfs = []
    for col in dfr.columns[len(df.columns):]:
        try:
            corr_dfs.append(get_correlations(dfr, col, group_cols, 'rows'))
        except TypeError:
            continue
    corr_df = pd.concat(corr_dfs, ignore_index=True)
    print(corr_df)

    print()
    print(f"Full hubness correlations (by {group_cols}):")
    corr_dfs = []
    for col in dff.columns[len(df.columns):]:
        try:
            corr_dfs.append(get_correlations(dff, col, group_cols, 'full'))
        except TypeError:
            continue
    corr_df2 = pd.concat(corr_dfs, ignore_index=True)
    print(corr_df2)
    corr_df = pd.concat([corr_df, corr_df2], ignore_index=True)
    corr_df.to_csv(outfile, index=False)
    print(f"Wrote results to {outfile}")


def get_correlations(df, col, group_cols, scope):
    records = []
    corrs = []
    uniques = [df[c].unique() for c in group_cols]
    for group_vals in itertools.product(*uniques):
        df_filtered = df.copy()
        cvs = list(zip(group_cols, group_vals))
        record = {'scope': scope, 'metric': col}
        for c, v in cvs:
            record[c] = v
            df_filtered = df_filtered[df_filtered[c] == v]
        record['pscore_test_corr'] = df_filtered['pscore_test'].corr(df_filtered[col])
        records.append(record)
    return pd.DataFrame.from_records(records)

if __name__ == '__main__':
    main(Path(sys.argv[1]))
