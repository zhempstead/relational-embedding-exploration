import argparse
from collections import defaultdict
import json
import pandas as pd
from tqdm import tqdm
import networkx as nx
import utils
import data_prep_utils as dpu
import textification.textify_relation as tr
from token_dict import TokenDict

num_threshold = 5000
distinct_threshold = 5


def clean_edges(record_dict):
    links = pd.DataFrame(record_dict.items(), columns=["token", "lst"])
    links['num'] = links["lst"].apply(len)
    links['distinct_num'] = links["lst"].apply(
        lambda x: len(set(map(lambda y: y.split("_")[0], x)))
    )
    links = links[(links["num"] != 1) & (links["num"] < num_threshold) & (
        links['distinct_num'] < distinct_threshold)]
    return links


def generate_graph(args):
    task = args.task
    suffix = "" if args.suffix == "" else "_" + args.suffix
    output_graph_path = "./graph/{}/{}{}.edgelist".format(task, task, suffix)
    output_dictionary_path = "./graph/{}/{}{}.dict".format(task, task, suffix)
    data_strategy_path = "./data/strategies/{}.txt".format(task)
    data_config_path = "./data/data_config.txt"
    with open(data_config_path, "r") as jsonfile:
        data_config = json.load(jsonfile)[task]
    with open(data_strategy_path, "r") as jsonfile:
        strategies = json.load(jsonfile)

    fs = utils.all_data_files_in_path(data_config["location"])
    edges = set()
    record_dict = defaultdict(list)
    record_dict2 = defaultdict(list)

    for path in tqdm(fs):
        table_name = path.split("/")[-1]
        table_strategy = strategies[table_name]
        filename = "".join(table_name.split(".")[:-1])
        df = pd.read_csv(path, encoding='latin1', sep=',', low_memory=False)
        # df.fillna(df.mode().iloc[0])
        columns = df.columns
        df = tr.quantize(df, table_strategy)
        # Add row edges
        for cell_value, row, col in tr._read_rows_from_dataframe(
                df, columns, table_strategy):
            grain_strategy = table_strategy[col]["grain"]
            decoded_row = dpu.encode_cell(row, grain=grain_strategy)
            decoded_value = dpu.encode_cell(cell_value, grain=grain_strategy)
            for value in decoded_value:
                for row in decoded_row:
                    row_name = "{}_row:{}".format(filename, row)
                    record_dict[value].append(row_name)
    
    record_dict = clean_edges(record_dict)
    for index, row in record_dict.iterrows():
        token, lst = row["token"], row["lst"]
        for row_id in lst:
            edges.add((token, row_id, 1.0 / len(lst)))
    # Save output graph and dictionary
    cc = TokenDict()
    with open(output_graph_path, "w") as f:
        for node_x, node_y, weight in iter(edges):
            decoded_x, decoded_y = cc.put(node_x), cc.put(node_y)
            f.write("{} {} {}\n".format(decoded_x, decoded_y, weight))
    cc.save(output_dictionary_path)
    print("Saved {} edges under{}".format(len(edges), output_graph_path))


if __name__ == "__main__":
    print("Generating graph for input")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='sample',
                        help='task to generate relation from')

    parser.add_argument('--suffix',
                        type=str,
                        default='',
                        help='a suffix to identify')

    args = parser.parse_args()
    generate_graph(args)
