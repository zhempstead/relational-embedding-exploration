import json
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import word2vec
import visualizer as VS
import eval_utils as EU
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import sys


# from keras.models import Sequential
# from keras import layers

embedding_storage = {
    "node2vec": '../node2vec/emb/',
    "ProNE": '../ProNE/emb/'
}

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]
test_size = embeddding_config["test_size"]

def classification_task_rf(X_train, X_test, y_train, y_test, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    def show_stats(model):
        pscore_train = accuracy_score(y_train, model.predict(X_train))
        pscore_test = accuracy_score(y_test, model.predict(X_test))
        print("Train accuracy {}, Test accuracy {}".format(
            pscore_train, pscore_test))
        return pscore_train, pscore_test

    return show_stats(rf)
    
def classification_task_logr(X_train, X_test, y_train, y_test, n_estimators=100):
    logr = LogisticRegression(penalty='l2', solver='liblinear')
    logr.fit(X_train, y_train)

    def show_stats(model):
        pscore_train = accuracy_score(y_train, model.predict(X_train))
        pscore_test = accuracy_score(y_test, model.predict(X_test))
        print("Train accuracy {}, Test accuracy {}".format(
            pscore_train, pscore_test))
        return pscore_train, pscore_test

    return show_stats(logr)


def classification_task_nn(task, X_train, X_test, y_train, y_test):
    input_size = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size,)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=['accuracy', tf.keras.metrics.CategoricalCrossentropy()]
    )

    model.fit(X_train, y_train, epochs=25)
    results = model.evaluate(X_test, y_test)
    print(results)


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
    trimmed_table = pd.read_csv(os.path.join(
        "../", location_processed), sep=',', encoding='latin')
    full_table = pd.read_csv(os.path.join(
        "../", location + target_file), sep=',', encoding='latin')

    Y = full_table[target_column]
    if args.task in ["kraken", "financial", "genes"]:
        Y = pd.Categorical(Y).codes

    # Set embeddings that are to be evaluated
    method = args.method
    all_embeddings_path = EU.all_files_in_path(
        embedding_storage[method], args.task)

    # Run through the embedding list and do evaluation
    for path in all_embeddings_path:
        model = KeyedVectors.load_word2vec_format(path)
        table_name = path.split("/")[-1][:-4]
        model_dict_path = "../graph/{}/{}.dict".format(args.task, table_name)
        print(model_dict_path)

        # Obtain textified & quantized data
        training_loss = [] 
        testing_loss = [] 
        for i in range(0, 100, 10):
            df_textified = EU.textify_df(
                trimmed_table, strategies, location_processed)
            x_vec = pd.DataFrame(
                EU.vectorize_df(
                    df_textified, model, model.vocab,
                    model_dict=model_dict_path, model_type=method
                )
            )
            model_2dim = EU.get_PCA_for_embedding(model)
            x_vec_2dim = pd.DataFrame(
                EU.vectorize_df(
                    df_textified, model_2dim, model.vocab,
                    model_dict=model_dict_path, model_type=method
                )
            )
            
            # Train a Random Forest classifier
            tests = train_test_split(x_vec, Y, test_size=0.1, random_state=10)
            train_loss, test_loss = classification_task_logr(*tests)
            training_loss.append(train_loss)
            testing_loss.append(test_loss)
        print(training_loss)
        print(testing_loss)
        # tests_2dim = train_test_split(x_vec_2dim, Y, test_size=test_size, random_state=10)
        # print("{}_dim:".format(model.vector_size))
        # for n_estimators in [10, 50, 100]:
        #     classification_task(*tests, n_estimators=n_estimators)
        # print("2_dim:")
        # for n_estimators in [10, 50, 100]:
        #     classification_task(*tests_2dim, n_estimators=n_estimators)
            
        # classification_task_nn(args.task, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    print("Evaluating results with word2vec model:")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='task to be evaluated on'
    )
    parser.add_argument(
        '--method',
        type=str,
        help='method of training'
    )

    args = parser.parse_args()

    print("Evaluating on task {}".format(args.task))
    evaluate_task(args)
