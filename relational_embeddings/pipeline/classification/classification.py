from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import tensorflow as tf

from relational_embeddings.lib.eval_utils import plot_tf_history, show_stats
from relational_embeddings.lib.utils import dataset_dir

def classification(cfg, outdir, indir=None):
    """
    Use model to produce embeddings on training and test data
    """
    if indir is None:
        indir = outdir.parent

    df_x = pd.read_csv(indir / 'embeddings.csv')
    df_y = pd.read_csv(dataset_dir(cfg.dataset.name) / 'base_y.csv')
    df_y = pd.Categorical(df_y[cfg.dataset.target_column]).codes

    df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(
        df_x, df_y, test_size=cfg.classification.test_size,
        random_state=cfg.classification.random_seed)

    with open(outdir / 'results.txt', 'w') as fout:
        for method in cfg.classification.methods:
            fout.write(f"Classification method '{method}':")
            method_func = METHOD2FUNC[method]
            method_func(df_train_x, df_test_x, df_train_y, df_test_y, cfg.classification, outdir, fout)


    print(f"Done with classification! Results at '{outdir}'")


def classification_task_rf(X_train, X_test, y_train, y_test, cfg, outdir, fout):
    rf = Pipeline([
        ("rf", RandomForestClassifier(random_state=7, min_samples_split=5))
    ])
    parameters = {
        'rf__n_estimators': [10, 20, 50, 70, 100, 250],
    }
    greg = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, verbose=0)
    greg.fit(X_train, y_train)
    return show_stats(greg, X_train, X_test, y_train, y_test, fout=fout)


def classification_task_logr(X_train, X_test, y_train, y_test, cfg, outdir, fout):
    lr = Pipeline([
        ("lr", LogisticRegression(random_state=7, penalty="elasticnet", solver="saga", max_iter=2000))
    ])
    parameters = {
        "lr__l1_ratio": [0.1, 0.3, 0.9, 1]
    }
    greg = GridSearchCV(estimator=lr, param_grid=parameters, cv=2, verbose=0)
    greg.fit(X_train, y_train)
    return show_stats(greg, X_train, X_test, y_train, y_test, fout=fout)


def classification_task_nn(X_train, X_test, y_train, y_test, cfg, outdir, fout):
    input_size = X_train.shape[1]
    ncategories = np.max(y_train) + 1
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size, )),
        tf.keras.layers.Dense(64, activation=tf.nn.sigmoid),
        # tf.keras.layers.Dense(32, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(ncategories, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        epochs=200,
                        verbose=0,
                        validation_data=(X_test, y_test))
    plot_tf_history(history, outdir / 'nn')
    model.evaluate(X_test, y_test, verbose=0)
    return show_stats(model, X_train, X_test, y_train, y_test, argmax=True, fout=fout)



METHOD2FUNC = {
    'nn': classification_task_nn,
    'logistic': classification_task_logr,
    'random_forest': classification_task_rf,
}
