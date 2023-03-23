import functools
import itertools
from pathlib import Path
import sys

import pandas as pd

INTERMEDIATE_FILE = "downstream_v_intrinsic_raw.csv"
IN_FILE = "downstream_v_intrinsic.csv"
OUT_FILE = "report.md"

CONTROL_COLS = ["dimensions", "model"]

REGRESSION = ["bio", "restbase"]

def main(parent_dir):
    df = gather_results(parent_dir)
    out = open(parent_dir / OUT_FILE, "w")
    out.write("# Downstream vs Intrinsic Rank Correlations\n")
    for conds in [
        {'dimensions': None, 'model': None},
        {'dimensions': 4, 'model': None},
        {'dimensions': 16, 'model': None},
        {'dimensions': None, 'model': 'graph2text_embdi'},
        {'dimensions': None, 'model': 'graph2text_leva'},
        {'dimensions': None, 'model': 'mf_sparse'},
        {'dimensions': None, 'model': 'mf_spectral'},
        {'dimensions': None, 'model': 'naive'},
    ]:
        notnull_conds = {k:v for k, v in conds.items() if v is not None}
        if not notnull_conds:
            out.write("## Overall\n")
        else:
            k, v = list(notnull_conds.items())[0]
            out.write(f"## Subset where {k} == {v}\n")
        out.write("### Single metric\n")
        single_results_table(df, conds).to_markdown(out, index=False)
        out.write("\n")
        out.write("### Two metrics\n")
        double_results_table(df, conds).to_markdown(out, index=False)
        out.write("\n")
    out.close()


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
        df['dataset'] = csv.parent.name
        if csv.parent.name in REGRESSION:
            df['spearman'] = df['spearman'] * -1
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    df = df[df['metric1'].str.endswith('ew')]
    df['metric1'] = df['metric1'].str[:-3]
    df = df[df['metric2'].isna() | df['metric2'].str.endswith('ew')]
    df.loc[~df['metric2'].isna(), 'metric2'] = df.loc[~df['metric2'].isna(), 'metric2'].str[:-3]
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
    main(Path(sys.argv[1]))
