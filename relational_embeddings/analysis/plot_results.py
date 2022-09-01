from pathlib import Path
import sys

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

NON_HP_COLUMNS = ['dataset', 'pscore_train', 'pscore_test']

def main(experiment_dir):
    infile = experiment_dir / 'results.csv'
    df = pd.read_csv(infile)
    hp_cols = list(set(df.columns) - set(NON_HP_COLUMNS))
    for i in range(len(hp_cols)):
        for j in range(i+1, len(hp_cols)):
            col1 = hp_cols[i]
            col2 = hp_cols[j]
            plot_hp_interaction(df, col1, col2, experiment_dir / f"{col1}-{col2}.png")
            
def plot_hp_interaction(df, col1, col2, outfile):
    df_grouped = df.groupby([col1, col2]).mean().reset_index()
    fig = plt.figure()
    new_plot = fig.add_subplot(projection='3d')

    if df_grouped[col1].dtype == object:
        cg = pd.Categorical(df_grouped[col1])
        categories = cg.categories
        df_grouped[col1] = cg.codes
        new_plot.set_xticks(df_grouped[col1].unique())
        new_plot.set_xticklabels(categories)
    if df_grouped[col2].dtype == object:
        cg = pd.Categorical(df_grouped[col2])
        categories = cg.categories
        df_grouped[col2] = cg.codes
        new_plot.set_yticks(df_grouped[col2].unique())
        new_plot.set_yticklabels(categories)


    new_plot.plot_trisurf(df_grouped[col1], df_grouped[col2], df_grouped['pscore_test'],
                    cmap='viridis', edgecolor='none');
    new_plot.set_xlabel(col1)
    new_plot.set_xticks(df_grouped[col1].unique())
    new_plot.set_ylabel(col2)
    new_plot.set_yticks(df_grouped[col2].unique())
    new_plot.set_zlabel('pscore_test')
    fig.savefig(outfile)
    plt.clf()

if __name__ == '__main__':
    main(Path(sys.argv[1]))
