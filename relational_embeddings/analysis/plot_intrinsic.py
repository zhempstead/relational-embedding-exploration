import math
from pathlib import Path
import sys

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

NON_HP_COLUMNS = ['dataset', 'pscore_train', 'pscore_test']

TASKS = ['embdi_MA', 'embdi_MR', 'as_MC', 'embdi_MC']
COMPOUND_TASKS = ['as_MC', 'embdi_MC']

def main(parent_experiment_dir):
    experiments = {}
    for experiment_dir in [d for d in parent_experiment_dir.iterdir() if d.is_dir()]:
        df = pd.read_csv(experiment_dir / 'results.csv')
        if 'text2model.dimensions' in df.columns:
            experiments[experiment_dir.name] = df.rename(columns={'text2model.dimensions': 'dimensions'})
        else:
            suffixes = df['model2emb.model_suffix'].unique()
            grouped = df.groupby('model2emb.model_suffix')
            for suffix in suffixes:
                df_suffix = grouped.get_group(suffix).drop('model2emb.model_suffix', axis=1)
                experiments[f"{experiment_dir.name}_{suffix}"] = df_suffix.rename(columns={'graph2model.dimensions': 'dimensions'})


    X = None
    for task in TASKS:
        data = {}
        for experiment, df in experiments.items():
            if X is None:
                X = sorted(df['dimensions'].unique())

            df['table'] = df['model'].str.split('.').str[0]

            filtered = df[df['model'].str.endswith(task)]
            tables = filtered['table'].unique()

            df = filtered.groupby(['dimensions', 'table']).mean().reset_index()
            grouped = df.sort_values('dimensions').groupby('table')
            for table in tables:
                if table not in data:
                    data[table] = {}
                data[table][experiment] = grouped.get_group(table).sort_values('dimensions')['pscore_test']

        num_plots = len(data)
        num_cols = 2
        num_rows = math.ceil(num_plots / 2)
        fig = plt.figure()
        for idx, (table, exp_data) in enumerate(data.items()):
            plot = fig.add_subplot(num_rows, num_cols, idx+1)
            for experiment, Y in exp_data.items():
                label = experiment if idx == 0 else None
                plot.plot(X, Y, label=label)
            plot.set_xscale('log')
            plot.set_xlabel('dimensions')
            plot.set_ylabel('accuracy')
            plot.set_title(table)
            plot.set_xticks(X, X)
            #plot.legend()
        fig.suptitle(task)
        fig.legend()
        fig.savefig(parent_experiment_dir / f"{task}.svg")

    data = {}
    for experiment, df in experiments.items():
        filtered = df[df["downstream.task"] == "classification"]
        data[experiment] = filtered.sort_values('dimensions')['pscore_test']
    import pdb; pdb.set_trace()
    fig = plt.figure()
    plot = fig.add_subplot()
    for exp, Y in data.items():
        plot.plot(X, Y, label=exp)
        plot.set_xscale('log')
        plot.set_xlabel('dimensions')
        plot.set_ylabel('accuracy')
        plot.set_xticks(X, X)
        plot.legend()
    fig.suptitle("Downstream classification accuracy")
    fig.savefig(parent_experiment_dir / "downstream.svg")




                


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
