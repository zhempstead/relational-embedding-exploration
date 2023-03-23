import functools
import itertools
from pathlib import Path
import sys

import pandas as pd

IN_FILE = "downstream_v_intrinsic_raw.csv"
OUT_FILE = "report.md"

CONTROL_COLS = ["dimensions", "model"]

REGRESSION = ["bio", "restbase"]

def main(parent_dir, hyperparam):
    full_df = gather_results(parent_dir)
    full_df = full_df[full_df['metric'].str.endswith('_ew')]
    full_df['metric'] = full_df['metric'].str[:-3]

    for (downstream, dataset, model), df in full_df.groupby(['downstream', 'dataset', 'model']):
        grouped = df.groupby([hyperparam, 'metric']).mean()
        pivot = grouped.reset_index().pivot(index=hyperparam, columns='metric', values='metric_score')
        pivot[downstream] = df.groupby(hyperparam).mean()['pscore_test']
        for col in pivot.columns:
            if col != 'regression':
                pivot[col] = pivot[col] / pivot[col].max()
        pivot.plot(title=f"{downstream} ({dataset}): {model}").get_figure().savefig(parent_dir / f'{downstream}_{dataset}_{model}.png')


def single_results_table(df, cond_dict):
    df = df_subset(df, {**cond_dict, 'metric2': None})
    dfp = df.pivot(index='metric1', columns='dataset', values='spearman')
    dfp = prettify(dfp)
    return dfp.reset_index()

def double_results_table(df, cond_dict):
    df = df_subset(df, cond_dict)
    df = df[~df['metric2'].isna()]
    dfp = df.pivot(index=['metric1', 'metric2'], columns='dataset', values='spearman')
    dfp = prettify(dfp)
    return dfp.reset_index()


def prettify(df):
    """
    - Convert float to string
    - Highlight max in each column
    """
    for col in df:
        colmax = df[col].max()
        argmax = (df[col] == colmax)
        df[col] = df[col].apply(lambda x: '{0:.3f}'.format(x))
        df.loc[argmax, col] = "**" + df.loc[argmax, col] + "**"
    return df


def df_subset(df, cond_dict):
    conds = [((df[col].isna()) if (val is None) else (df[col] == val)) for col, val in cond_dict.items()]
    all_conds = True
    [all_conds := (all_conds & cond) for cond in conds]
    return df[all_conds].drop(columns=cond_dict.keys())

def gather_results(parent_dir):
    dfs = []
    for csv in parent_dir.glob(f'*/{IN_FILE}'):
        df = pd.read_csv(csv)
        df['downstream'] = csv.parent.name
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    return df

def bold_formatter(x, value):
    """Format a number in bold when identical to a given value.
    
    Args:
        x: Input number.
        
        value: Value to compare x with.

    Returns:
        String converted output.

    """
    if x == value:
        return f"\\textbf{{{x}}}"
    else:
        return str(x)

if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2])
