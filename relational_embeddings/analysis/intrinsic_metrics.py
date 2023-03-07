import itertools
from pathlib import Path
import sys

import pandas as pd

def main(parent_dir):
    experiment_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    exp_dfs = []
    for experiment_dir in experiment_dirs:
        results = [r for r in experiment_dir.rglob('results.csv')]
        dfs = []
        for result in results:
            metrics = result.parent / 'intrinsic' / 'intrinsic.csv'
            df_r = pd.read_csv(result)
            df_m = pd.read_csv(metrics)
            df_m = df_m[['metric', 'successes', 'total']].groupby('metric').sum().reset_index()
            df_m['metric_score'] = df_m['successes'] / df_m['total']
            
            # Hack for cross join
            df_r['key'] = 1
            df_m['key'] = 1
            df = df_r.merge(df_m).drop(columns=['model', 'pscore_train', 'successes', 'total', 'key'])
            dfs.append(df)
        exp_df = pd.concat(dfs)
        exp_df['model'] = experiment_dir.name
        for col in exp_df.columns:
            if 'dimensions' in col:
                exp_df = exp_df.rename(columns={col: 'dimensions'})
            if col in ['table2graph.method', 'model2emb.model_suffix']:
                exp_df['model'] = exp_df['model'] + '_' + exp_df[col]
                exp_df = exp_df.drop(columns=[col])
        exp_dfs.append(exp_df)
    df = pd.concat(exp_dfs)
    print_corrs(df)
    for control_cols in [['dimensions'], ['model'], ['dimensions', 'model']]:
        print(f"Control cols: {control_cols}")
        for vals, df_subset in df.groupby(control_cols):
            print(f"  ...for {vals}:")
            print_corrs(df_subset)


def print_corrs(df, control_cols=[]):
    """
    df should have columns [metric, metric_score, pscore_test]
    """
    metrics = df['metric'].unique()

    for num in range(1, len(metrics)+1):
        max_corr = -1
        best_subset = None
        for ms in itertools.combinations(metrics, num):
            df_subset = df[df['metric'].isin(ms)].drop(columns=['metric'])
            df_subset = df_subset.groupby(list(df_subset.columns[df_subset.columns != 'metric_score'])).mean().reset_index()
            rank_corr = df_subset[['pscore_test', 'metric_score']].corr('spearman').iloc[0, 1]
            if rank_corr > max_corr:
                max_corr = rank_corr
                best_subset = ms
        print(f"For {num} metric(s) the best correlation is {max_corr}: {best_subset}")


if __name__ == '__main__':
    main(Path(sys.argv[1]))
