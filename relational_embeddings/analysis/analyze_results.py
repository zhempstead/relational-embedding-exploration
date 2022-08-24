from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


IGNORE_COLUMNS = ['dataset', 'pscore_train']

def gridsearch(results_df, iter_columns):
    if iter_columns:
        for iter_col_vals, df in results_df.groupby(iter_columns):
            coef_df = get_coef_df(df.drop(columns=iter_columns))
            if len(iter_columns) > 1:
                print(list(zip(iter_columns, iter_col_vals)), 'results:')
            else:
                print(iter_columns[0], iter_col_vals, 'results:')
            print(coef_df)
    else:
        coef_df = get_coef_df(results_df)
        print(coef_df)

def get_coef_df(df):
    X = df.drop(columns=IGNORE_COLUMNS + ['pscore_test'])
    y = df['pscore_test']
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_interact = poly.fit_transform(X)
    X_interact = X_interact / X_interact.max(axis=0)
    clf = LinearRegression()
    clf.fit(X_interact, y)
    coef_df = pd.DataFrame({'coef': poly.get_feature_names_out(), 'val': clf.coef_})
    coef_df['sort_col'] = coef_df['val'].abs()
    coef_df = coef_df.sort_values('sort_col', ascending=False).drop(columns=['sort_col'])
    return coef_df


def main(experiment_dir, *iter_columns):
    infile = experiment_dir / 'results.csv'
    df = pd.read_csv(infile)
    iter_columns = list(iter_columns)
    for col in df.columns:
        if df[col].dtype == object and col not in iter_columns and col not in IGNORE_COLUMNS:
            iter_columns.append(col)
    gridsearch(df, iter_columns)

if __name__ == '__main__':
    main(Path(sys.argv[1]), *sys.argv[2:])
