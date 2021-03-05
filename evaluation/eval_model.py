import json
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import argparse
import os
#import word2vec
import visualizer as VS
import eval_utils as EU
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import sys
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

embedding_storage = {"node2vec": '../node2vec/emb/', "ProNE": '../ProNE/emb/'}

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]
test_size = embeddding_config["test_size"]


def evaluate_task(args):
    # Load task config information
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)
    with open("../data/strategies/" + args.task + ".txt", "r") as jsonfile:
        strategies = json.load(jsonfile)
    config = data_config[args.task]
    location = config["location"]
    target_file = config["target_file"]
    location_processed = config["location_processed"]
    target_column = config["target_column"]

    # Load data
    trimmed_table = pd.read_csv(os.path.join("../", location_processed),
                                sep=',',
                                encoding='latin')
    full_table = pd.read_csv(os.path.join("../", location + target_file),
                             sep=',',
                             encoding='latin')
   
    Y = full_table[target_column]
    if args.task in ["kraken", "financial", "gene"]:
        Y = pd.Categorical(Y).codes

    # Set embeddings that are to be evaluated
    method = args.method
    all_embeddings_path = EU.all_files_in_path(embedding_storage[method],
                                               args.task)

    # Run through the embedding list and do evaluation
    for path in all_embeddings_path:
        model = KeyedVectors.load_word2vec_format(path)
        table_name = path.split("/")[-1][:-4]
        if "_sparse" in table_name or "_spectral" in table_name:
            table_name = "_".join(table_name.split("_")[:-1])
        model_dict_path = "../graph/{}/{}.dict".format(args.task, table_name)
        print(model_dict_path)

        # Obtain textified & quantized data
        training_loss = []
        testing_loss = []
        df_textified = EU.textify_df(trimmed_table, strategies,
                                     location_processed)
        x_vec = EU.vectorize_df(df_textified,
                                model,
                                model.vocab,
                                model_dict=model_dict_path,
                                model_type=method)

        for i in range(50, 100, 20):
            model_2dim = EU.get_PCA_for_embedding(model, ndim=i)
            x_vec_2dim = EU.vectorize_df(df_textified,
                                         model_2dim,
                                         model.vocab,
                                         model_dict=model_dict_path,
                                         model_type=method)
            print(x_vec_2dim.shape, Y.shape)
            x_vec_2dim['Y'] = Y
            x_vec_2dim = x_vec_2dim.fillna(0)
            #x_vec_2dim = x_vec_2dim.dropna(axis=0, how='any')
            Y = x_vec_2dim['Y']
            x_vec_2dim =  x_vec_2dim.drop('Y', axis = 1)
            print(x_vec_2dim.shape, Y.shape)
            tests = train_test_split(x_vec_2dim,
                                     Y,
                                     test_size=test_size,
                                     random_state=10)
            #train_loss, test_loss = 
            EU.lassoRegression(*tests)
            #training_loss.append(train_loss)
            #testing_loss.append(test_loss[0])
        #     print(train_loss)
        #     print(test_loss)
        # print("LR Train Loss: {}, Test Loss: {}".format(train_loss, test_loss))

    

def simple_regression(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    print("LR Train score: {}, Test score: {}".format(train_score, test_score))

if __name__ == "__main__":
    print("Evaluating results with word2vec model:")

    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        required=True,
                        help='task to be evaluated on')
    parser.add_argument('--method', type=str, help='method of training')

    args = parser.parse_args()

    print("Evaluating on task {}".format(args.task))
    evaluate_task(args)
