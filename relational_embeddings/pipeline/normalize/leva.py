from collections import defaultdict
import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from relational_embeddings.lib.utils import all_csv_in_path

DATE_PATTERN = re.compile("(\d+/\d+/\d+)")


def leva_normalize(indir, outdir, cfg):
    """
    Normalize each table based on leva:
    - Quantize numerical columns
    - Lower case, remove punctuation
    - Either single or multiple tokens per value depending on column grain size
    """
    strategies = dict()
    for infile in all_csv_in_path(indir):
        df = pd.read_csv(infile, sep=",", low_memory=False)
        df = filter_target_col(df, cfg.target_column)

        strategy = get_strategy(df)
        strategies[infile.name] = strategy

        df = normalize_df(df, strategy, cfg)

        df.to_csv(outdir / infile.name, index=False)

    # Write strategies
    with open(outdir / "strategy.txt", "w") as json_file:
        json.dump(strategies, json_file, indent=4)


def filter_target_col(df, target_col):
    if target_col in df.columns:
        return df.drop(target_col, axis=1)


def get_strategy(df):
    """
    Return strategy dict for given input df
    """
    strategy = defaultdict(dict)

    for col in df.columns:
        integer_strategy, grain_strategy = "augment", "cell"
        convert_dt = is_dt_col(df[col])
        num_distinct_numericals = df[col].nunique()

        if "id" not in col and (convert_dt or df[col].dtype in [
            np.float,
            np.float16,
            np.float32,
            np.float64,
        ]):
            if convert_dt:
                strategy[col]["convert_dt"] = True
                # This gets done redundantly later. This redundancy lets get_strategy() remain a
                # pure function
                floatcol = dt_col_to_float(df[col])
            else:
                floatcol = df[col]

            if abs(floatcol.skew()) >= 2:
                integer_strategy = "eqd_quantize"
            else:
                integer_strategy = "eqw_quantize"

        if df[col].dtype in [np.int64, np.int32, np.int64, np.int]:
            if df[col].max() - df[col].min() >= 5 * df[col].shape[0]:
                if abs(df[col].skew()) >= 2:
                    integer_strategy = "eqd_quantize"
                else:
                    integer_strategy = "eqw_quantize"

        if df[col].dtype == np.object:
            num_tokens_med = (df[col].str.count(" ") + 1).median()
            if num_tokens_med >= 10:
                grain_strategy = "token"

        strategy[col]["int"] = integer_strategy
        strategy[col]["grain"] = grain_strategy
    return strategy


def is_dt_col(series):
    '''
    Heuristic is that if we *can't* convert a column to float without error but we *can* call
    pd.to_datetime without error the column is a date/datetime.
    '''
    try:
        floatcol = series.astype(float)
        return False
    except (TypeError, ValueError):
        try:
            dtcol = pd.to_datetime(series)
            return True
        except (TypeError, ValueError):
            return False


def normalize_df(df, strategy, cfg):
    df = quantize(df, strategy, cfg)
    for col in df.columns:
        lowercase_removepunct(df, col)
        grain = strategy[col]["grain"]
        if grain == "cell":
            df[col] = df[col].str.replace(" ", "_")
    return df


def quantize(df, strategy, cfg):
    num_bins = cfg.num_bins
    # Not enough numerical values for binning
    if df.shape[0] < 2 * num_bins:
        return df

    bin_percentile = 100.0 / num_bins
    for col in df.columns:
        if strategy[col].get('convert_dt'):
            df[col] = dt_col_to_float(df[col])

        if df[col].dtype not in [
            np.int64,
            np.int32,
            np.int64,
            np.float,
            np.int,
            np.float16,
            np.float32,
            np.float64,
        ]:
            continue

        augment = True

        if strategy[col]["int"] == "skip":
            df.drop(col, axis=1, inplace=True)
            continue
        if strategy[col]["int"] == "stringify":
            quantized_col = df[col]
            augment = False
        if strategy[col]["int"] == "augment":
            quantized_col = df[col]
        if strategy[col]["int"] == "eqd_quantize":
            bins = [np.percentile(df[col], i * bin_percentile) for i in range(num_bins)]
            quantized_col = pd.Series(np.digitize(df[col], bins))
        if strategy[col]["int"] == "eqw_quantize":
            # The '[:-1]' prevents the max value from being converted to num_bins+1
            bins = np.histogram_bin_edges(df[col].dropna(), bins=num_bins)[:-1]
            quantized_col = pd.Series(np.digitize(df[col], bins))

        quantized_col = quantized_col.astype(str)

        if augment:
            sanitized_col = sanitize_col(col)
            # Special symbol to tell apart augmentation from space
            quantized_col = sanitized_col + "_<#>_" + quantized_col.astype(str)

        # Values that were originally null get put in the highest bin
        # Instead, set them to None
        quantized_col[pd.isnull(df[col])] = None

        df[col] = quantized_col
    return df


def dt_col_to_float(series):
    '''
    Convert into float (seconds from epoch).
    We treat date/datetimes like numeric columns and don't take advantage of implicit joins
    '''
    dtcol = pd.to_datetime(series)
    return (dtcol - dt.datetime(1970, 1, 1)).dt.total_seconds()


def lowercase_removepunct(df, col):
    """
    Also normalizes null values to None
    """
    df[col] = df[col].astype(str)

    df.loc[pd.isnull(df[col]), col] = None
    df[col] = df[col].replace("", None)
    df[col] = df[col].replace("\\N", None)
    # Filter out dates, for some reason
    df.loc[df[col].str.match(DATE_PATTERN), col] = None

    df[col] = df[col].str.lower()
    df[col] = df[col].str.replace(",", " ")
    df[col] = df[col].str.replace("  ", " ")
    df[col] = df[col].str.strip()

def sanitize_col(col):
    return str(col).lower().strip().replace(" ", "_")


if __name__ == "__main__":
    tokenize()
