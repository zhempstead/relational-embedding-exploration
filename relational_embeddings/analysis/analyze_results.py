import itertools
from pathlib import Path
import sys

import pandas as pd

IGNORE_COLUMNS = ['dataset', 'pscore_train', 'pscore_test']

def main(experiment_dir):
    infile = experiment_dir / 'results.csv'
    df = pd.read_csv(infile)
    print_avgs(df)


def print_avgs(df):
    iter_columns = [col for col in df.columns if col not in IGNORE_COLUMNS]
    print("SINGLE VARIABLE:")
    for col in iter_columns:
        print(f"{col} averages:")
        print(df.groupby(col).mean()['pscore_test'].sort_values(ascending=False))
        print(f"{col} maxima:")
        print(df.groupby(col).max()['pscore_test'].sort_values(ascending=False))
    print("DOUBLE VARIABLE:")
    for col1, col2 in itertools.combinations(iter_columns, 2):
        print(f"{col1} and {col2} averages:")
        print(df.groupby([col1, col2]).mean()['pscore_test'].sort_values(ascending=False))
        print(f"{col1} and {col2} maxima:")
        print(df.groupby([col1, col2]).max()['pscore_test'].sort_values(ascending=False))
    

if __name__ == '__main__':
    main(Path(sys.argv[1]))
