import argparse
import csv
import os
import pandas as pd
from os import listdir
from collections import defaultdict
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import json
import random

import data_prep_utils as dpu

RELATION_STRATEGY = ['row_and_col', 'row', 'col', 'alex']
INTEGER_STRATEGY = ['skip', 'stringify', 'augment', "quantize"]
TEXTIFICATION_GRAIN = ['cell', 'token']
OUTPUT_FORMAT = ['sequence_text', 'windowed_text']

def _read_rows_from_dataframe(df, columns, strategies):
    for index, el in df.iterrows():
        for c in columns:
            integer_strategy = strategies[c]
            cell_value = el[c]

            if (cell_value == "\\N") or (cell_value == ""):
                continue
            # We check the cell value is valid before continuing
            if not dpu.valid_cell(cell_value):
                continue
            # If strategy is skip, we check that first
            if df[c].dtype in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
                cell_value = str(c) + "_<#>_" + str(el[c])  # special symbol to tell apart augmentation from space
            yield cell_value, index, c

def _read_columns_from_dataframe(df, columns, strategies):
    for c in columns:
        integer_strategy = strategies[c]

        # If strategy is skip, we check that first
        if df[c].dtype in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
            if integer_strategy == 'skip':
                continue  # no numerical columns
            data_values = df[c]
            for cell_value in data_values:
                # We check the cell value is valid before continuing
                if not dpu.valid_cell(cell_value):
                    continue
                if integer_strategy == 'stringify':
                    cell_value = str(cell_value)
                elif integer_strategy == 'augment':
                    cell_value = str(c) + "_<#>_" + str(cell_value)  # special symbol to tell apart augmentation from space
                yield cell_value, c
        else:
            data_values = df[c]
            for cell_value in data_values:
                if (cell_value == "\\N"):
                    continue
                # We check the cell value is valid before continuing
                if not dpu.valid_cell(cell_value):
                    continue
                yield cell_value, c

def quantize(df, strategies):
    try:
        with open("../embedding_config.json", "r") as jsonfile:
            embeddding_config = json.load(jsonfile)
    except FileNotFoundError:
        with open("./embedding_config.json", "r") as jsonfile:
            embeddding_config = json.load(jsonfile)

    num_bins = embeddding_config["num_bins"]
    # Not enough numerical values for binning
    if df.shape[0] < 2 * num_bins:
        return df 

    bin_percentile = 100.0 / num_bins
    for col in df.columns:
        if df[col].dtype not in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
            continue 
        
        if strategies[col]["int"] == "eqw_quantize":
            bins = [np.percentile(df[col], i * bin_percentile) for i in range(num_bins)]
            df[col] = np.digitize(df[col], bins)
        if strategies[col]["int"] == "eqh_quantize":
            bins = [i * (df[col].max() - df[col].min()) / num_bins for i in range(num_bins)]
            df[col] = np.digitize(df[col], bins)
    return df

def serialize_row_and_column(paths, output_file, integer_strategy=None, grain=None, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    with open("./data/strategies/genes.txt", "r") as jsonfile:
        strategies = json.load(jsonfile)
    for path in tqdm(paths):
        if debug:
            print(str(current) + "/" + str(total) + ":" + path)
            current += 1
        
        if "schema" in path or "json" in path: 
            continue 
        table_name = path.split("/")[-1]
        table_strategy = strategies[table_name]
        df = pd.read_csv(path, encoding='latin1', sep=',')
        df = quantize(df, table_strategy)
        columns = df.columns 
        # Check if relation is valid. Otherwise skip to next
        # if not dpu.valid_relation(df):
            # continue
        with open(output_file, 'a') as f:
            for cell_value, row, col in _read_rows_from_dataframe(
                df, columns, table_strategy):
                grain_strategy = table_strategy[col]["grain"]
                decoded_row = dpu.encode_cell(row, grain=grain_strategy)
                decoded_value = dpu.encode_cell(cell_value, grain=grain_strategy)
                for cv in decoded_value:
                    f.write(" " + cv)
            # Columns
            # for cell_value in _read_columns_from_dataframe(df, df.columns, integer_strategy=integer_strategy):
            #     values = dpu.encode_cell(cell_value, grain=grain)
            #     for cv in values:
            #         f.write(" " + cv)
    f.close()

def alex__read_rows_from_dataframe(df, columns, integer_strategy='skip'):
    for index, el in df.iterrows():
        for c in columns:
            cell_value = el[c]
            if (cell_value == "\\N"):
                continue
            # We check the cell value is valid before continuing
            if not dpu.valid_cell(cell_value):
                continue
            # If strategy is skip, we check that first
            if df[c].dtype in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
                if integer_strategy == 'skip':
                    continue  # no numerical columns
                elif integer_strategy == 'stringify':
                    cell_value = str(el[c])
                elif integer_strategy == 'augment':
                    cell_value = str(c) + "_<#>_" + str(el[c])  # special symbol to tell apart augmentation from space
            yield cell_value, c, index

def alex__serialize_row_col(paths, output_file, integer_strategy=None, grain=None, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in tqdm(paths):
        if debug:
            print(str(current) + "/" + str(total))
            current += 1
        df = pd.read_csv(path, encoding='latin1', sep=',')
        df = quantize(df, excluding = ["event_id", "result"])

        columns = df.columns
        with open(output_file, 'a') as f:
            # Rows
            for _ in range(20):
                df = df.iloc[np.random.permutation(len(df))]
                df = df[np.random.permutation(df.columns)]
                for cell_value, c, index in alex__read_rows_from_dataframe(df, df.columns, integer_strategy=integer_strategy):
                    values = dpu.encode_cell((cell_value, c), grain=grain)
                    for cv in values:
                        f.write(" " + cv)
    
    f.close()

def read_rows_values(path, integer_strategy, grain, dataframe=None):
    """
    Returns a map (row index -> [values]
    :param path:
    :param integer_strategy:
    :param grain:
    :param dataframe:
    :return:
    """
    row_values = defaultdict(list)
    df = dataframe
    if df is None:
        df = pd.read_csv(path, encoding='latin1', sep=',')
        # Check if relation is valid. Otherwise skip to next
        if not dpu.valid_relation(df):
            return None
    columns = df.columns

    # Rows
    for cell_value, index in _read_rows_from_dataframe(df, columns, integer_strategy=integer_strategy):
        # If valid, we clean and format it and return it
        values = dpu.encode_cell(cell_value, grain=grain)
        for cv in values:
            row_values[index].append(cv)
    return row_values


def serialize_row(paths, output_file, integer_strategy=None, grain=None, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in tqdm(paths):
        if debug:
            print(str(current) + "/" + str(total))
            current += 1

        # map_rowidx_values = read_rows_values(path, integer_strategy, grain)
        # values = []
        # # unpack map into list of values
        # for index, values in map_rowidx_values.items():
        #     values.extend(values)
        # with open(output_file, 'a') as f:
        #     # for rowv in values:
        #     data_string = " ".join(values)
        #     f.write(data_string)

        if "schema" in path or "json" in path:
            continue

        df = pd.read_csv(path, encoding='latin1', sep=',')
        df = quantize(df, excluding = ["event_id", "result"])
        # Check if relation is valid. Otherwise skip to next
        # if not dpu.valid_relation(df):
        #     continue
        columns = df.columns
        
        with open(output_file, 'a') as f:
            for _ in range(20):
                df = df.iloc[np.random.permutation(len(df))]
                df = df[np.random.permutation(df.columns)]
                # Rows
                for cell_value in _read_rows_from_dataframe(df, df.columns, integer_strategy=integer_strategy):
                    # If valid, we clean and format it and return it
                    values = dpu.encode_cell(cell_value, grain=grain)
                    for cv in values:
                        f.write(" " + cv)


def read_column_values(path, integer_strategy, grain, dataframe=None):
    """
    Reads relation in path, returns a map (column -> [values])
    :param path:
    :param integer_strategy:
    :param grain:
    :return:
    """
    to_return = defaultdict(list)
    df = dataframe
    if df is None:
        df = pd.read_csv(path, encoding='latin1', sep=',')
        # Filtering out non-valid relations
        if not dpu.valid_relation(df):
            return None
    columns = df.columns
    # Columns
    for cell_value, c in _read_columns_from_dataframe(df, columns, integer_strategy=integer_strategy):
        values = dpu.encode_cell(cell_value, grain=grain)
        for cv in values:
            # to_return.append(cv)
            to_return[c].append(cv)
    return to_return


def serialize_column(paths, output_file, integer_strategy=None, grain=None, debug=False):
    try:
        os.remove(output_file)
    except FileNotFoundError:
        print("Creating new file for writing data")

    total = len(paths)
    current = 0
    for path in tqdm(paths):
        if debug:
            print(str(current) + "/" + str(total))
            current += 1

        # map_column_values = read_column_values(path, integer_strategy, grain)
        # # we need a list of values only, so we unpack the dictionary
        # values = []
        # for k, v in map_column_values.items():
        #     values.extend(v)
        # with open(output_file, 'a') as f:
        #     data_string = " ".join(map_column_values)
        #     f.write(data_string)

        if "schema" in path or "json" in path:
            continue

        df = pd.read_csv(path, encoding='latin1', sep=',')
        df = quantize(df, excluding = ["event_id", "result"])

        # Filtering out non-valid relations
        # if not dpu.valid_relation(df):
        #     continue
        columns = df.columns
        with open(output_file, 'a') as f:
            for _ in range(20):
                df = df.iloc[np.random.permutation(len(df))]
                df = df[np.random.permutation(df.columns)]
                # Columns
                for cell_value in _read_columns_from_dataframe(df, df.columns, integer_strategy=integer_strategy):
                    values = dpu.encode_cell(cell_value, grain=grain)
                    for cv in values:
                        f.write(" " + cv)


def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store" and f != "base_processed.csv"]
    return fs

def main(args):
    path = args.dataset
    relation_strategy = args.relation_strategy
    integer_strategy = args.integer_strategy
    grain = args.grain
    output = args.output
    debug = args.debug
    output_format = args.output_format
    fs = all_files_in_path(path)
    if output_format == "sequence_text":
        if relation_strategy == "row":
            serialize_row(fs, output, integer_strategy=integer_strategy, grain=grain, debug=debug)
        elif relation_strategy == "col":
            serialize_column(fs, output, integer_strategy=integer_strategy, grain=grain, debug=debug)
        elif relation_strategy == "row_and_col":
            serialize_row_and_column(fs, output, integer_strategy=integer_strategy, grain=grain, debug=debug)
        elif relation_strategy == "alex":
            alex__serialize_row_col(fs, output, integer_strategy=integer_strategy, grain=grain, debug=debug)
        else:
            print("Mode not supported. <row, col, row_and_col>")
        
    print(output)
    print("Done!")


if __name__ == "__main__":
    print("Textify relation")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to collection of relations')
    parser.add_argument('--relation_strategy', default='row_and_col', type=str, choices=RELATION_STRATEGY,
                        help='Strategy to capture relationships, row, col, or row_and_col')
    parser.add_argument('--integer_strategy', default='stringify', type=str, choices=INTEGER_STRATEGY,
                        help='strategy to determine how to deal with integers')
    parser.add_argument('--grain', default='cell', type=str, choices=TEXTIFICATION_GRAIN,
                        help='choose whether to process individual tokens (token), or entire cells (cell)')
    parser.add_argument('--output', type=str, default='textified.txt', help='path where to write the output file')
    parser.add_argument('--output_format', default='sequence_text', choices=OUTPUT_FORMAT,
                        help='sequence_text or windowed_text')
    parser.add_argument('--debug', default=False, help='whether to run program in debug mode or not')

    args = parser.parse_args()

    main(args)
