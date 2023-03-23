import itertools
from pathlib import Path
import sys

import pandas as pd

INTERMEDIATE_FILE = "downstream_v_intrinsic_raw.csv"
OUT_FILE = "downstream_v_intrinsic.csv"

CONTROL_COLS = ["dimensions", "model"]

def main(parent_dir):
    main_file = parent_dir / INTERMEDIATE_FILE
    if not main_file.exists():
        df = gather_results(parent_dir)
        df.to_csv(main_file, index=False)
    else:
        df = pd.read_csv(main_file)

    print(f"pscore_test ranges from {df['pscore_test'].min()} to {df['pscore_test'].max()}")
    res_dfs = []
    for ccols in powerset(CONTROL_COLS):
        if not ccols:
            res_df = corrs(df)
            for ccol in CONTROL_COLS:
                res_df[ccol] = None
            res_dfs.append(res_df)
            continue
        for vals, df_subset in df.groupby(list(ccols)):
            res_df = corrs(df_subset)
            for ccol in CONTROL_COLS:
                res_df[ccol] = None
            if len(ccols) == 1:
                res_df[ccols[0]] = vals
            else:
                for idx in range(len(ccols)):
                    res_df[ccols[idx]] = vals[idx]
            res_dfs.append(res_df)

    res_df = pd.concat(res_dfs)
    res_df.to_csv(parent_dir / OUT_FILE, index=False)


def gather_results(parent_dir):
    experiment_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    exp_dfs = []
    for experiment_dir in experiment_dirs:
        results = [r for r in experiment_dir.rglob('results.csv')]
        dfs = []
        for result in results:
            metrics = result.parent / 'intrinsic' / 'intrinsic.csv'
            df_r = pd.read_csv(result)
            df_m = pd.read_csv(metrics)
            df_m['ew'] = df_m['successes'] / df_m['total']
            df_m = df_m[['metric', 'successes', 'total', 'ew']]
            df_m = df_m.groupby('metric').agg({'successes': 'sum', 'total': 'sum', 'ew': 'mean'}).reset_index()
            df_m['sum'] = df_m['successes'] / df_m['total']
            df_m = df_m[['metric', 'ew', 'sum']]
            df_m = df_m.set_index('metric').stack().reset_index()
            df_m['metric'] = df_m['metric'] + '_' + df_m['level_1']
            df_m = df_m.rename(columns={0: 'metric_score'})
            df_m = df_m[['metric', 'metric_score']]
            
            # Hack for cross join
            df_r['key'] = 1
            df_m['key'] = 1
            df = df_r.merge(df_m).drop(columns=['model', 'pscore_train', 'key'])
            dfs.append(df)
        exp_df = pd.concat(dfs)
        exp_df['model'] = experiment_dir.name
        for col in exp_df.columns:
            if col in ['table2graph.method', 'model2emb.model_suffix', 'downstream.model_suffix']:
                exp_df['model'] = exp_df['model'] + '_' + exp_df[col]
                exp_df = exp_df.drop(columns=[col])
            if '.' in col:
                exp_df = exp_df.rename(columns={col: col.split('.')[-1]})
        exp_dfs.append(exp_df)
    return pd.concat(exp_dfs)

def corrs(df):
    """
    df should have columns [metric, metric_score, pscore_test]
    """
    results = {'spearman': [], 'metric1': [], 'metric2': []}
    metrics = df['metric'].unique()
    for num in range(1, 3):
        for ms in itertools.combinations(metrics, num):
            df_subset = df[df['metric'].isin(ms)].drop(columns=['metric'])
            df_subset = df_subset.groupby(list(df_subset.columns[df_subset.columns != 'metric_score'])).mean().reset_index()
            rank_corr = df_subset[['pscore_test', 'metric_score']].corr('spearman').iloc[0, 1]
            results['metric1'].append(ms[0])
            if len(ms) > 1:
                results['metric2'].append(ms[1])
            else:
                results['metric2'].append(None)
            results['spearman'].append(rank_corr)
    return pd.DataFrame(results)


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

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

if __name__ == '__main__':
    main(Path(sys.argv[1]))
