from pathlib import Path
import sys

from omegaconf import OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

NON_HP_COLUMNS = ['dataset', 'pscore_train', 'pscore_test']

def main(parent_dir):
    exp_dirs = sorted([d for d in parent_dir.glob('*') if d.is_dir()])
    cfg = OmegaConf.load(parent_dir / 'plot_cfg.yaml')

    dfs = []
    for exp_dir in exp_dirs:
        infile = exp_dir / 'results.csv'
        df = pd.read_csv(infile)
        df = df.drop(columns='model')
        df['model'] = exp_dir.name
        dfs.append(df)

    separate_plot_calls = cfg.get('separate_plot_calls', False)
    
    df = pd.concat(dfs)
    if cfg.filter_cols:
        for col, val in cfg.filter_cols.items():
            if col not in df.columns:
                continue
            df = df[df[col] == val]

    # DELETE ME
    df.loc[pd.isnull(df['corpus_size']), 'corpus_size'] = df['corpus_size'].min()

    if cfg.plot_cols:
        for plot_vals, df in tuple(df.groupby(list(cfg.plot_cols))):
            title = cfg.title + f", {cfg.plot_cols} = {plot_vals}"
            plot(df, cfg, title, parent_dir / f"plot_{plot_vals}.png")
    else:
        plot(df, cfg, cfg.title, parent_dir / "plot.png")

def plot(df, cfg, title, outfile):
    pivot = df.pivot(index=cfg.xcol, columns=cfg.line_cols, values=cfg.ycol)
    pivot = pivot.fillna(method='ffill')
    pivot.plot(logx=cfg.get('logx', False)).get_figure().savefig(outfile)

def splitSerToArr(ser):
    return [ser.index, ser.as_matrix()]


if __name__ == '__main__':
    main(Path(sys.argv[1]))
