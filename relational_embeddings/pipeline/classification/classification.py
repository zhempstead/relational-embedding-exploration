from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import tensorflow as tf

from relational_embeddings.lib.eval_utils import plot_tf_history, report_metric
from relational_embeddings.lib.utils import dataset_dir

def classification(cfg, outdir, indir=None):
    """
    Use model to produce embeddings on training and test data
    """
    if indir is None:
        indir = outdir.parent

    random_state = np.random.RandomState(seed=cfg.classification.random_seed)

    df_x = pd.read_csv(indir / 'embeddings.csv')
    df_y = pd.read_csv(dataset_dir(cfg.dataset.name) / 'base_y.csv')

    shuffled_idx = random_state.permutation(len(df_x))
    df_x = df_x.reindex(shuffled_idx)
    df_y = df_y.reindex(shuffled_idx)
    df_y = pd.Categorical(df_y[cfg.dataset.target_column]).codes

    
    kfold = StratifiedKFold(n_splits=cfg.classification.cv_splits)

    train_test_idxs = list(kfold.split(df_x, df_y))

    pscore_train_avgs = []
    pscore_test_avgs = []

    with open(outdir / 'results.txt', 'w') as fout:
        for method in cfg.classification.methods:
            tee(fout, f"Classification method '{method}':")
            method_func = METHOD2FUNC[method]
            pscore_trains = []
            pscore_tests = []
            for it in range(cfg.classification.cv_splits):
                tee(fout, f"  Iteration {it+1}/{cfg.classification.cv_splits}:")
                train_idx, test_idx = train_test_idxs[it]
                df_train_x = df_x.iloc[train_idx]
                df_test_x = df_x.iloc[test_idx]
                df_train_y = df_y[train_idx]
                df_test_y = df_y[test_idx]
                pscore_train, pscore_test, cm, y_sample, pred_sample = \
                    method_func(df_train_x, df_test_x, df_train_y, df_test_y, cfg.classification, outdir)
                pscore_trains.append(pscore_train)
                pscore_tests.append(pscore_test)
                tee(fout, f"    real: {y_sample}")
                tee(fout, f"    pred: {pred_sample}")
                tee(fout, f"    Train accuracy {pscore_train}, Test accuracy {pscore_test}")
                tee(fout, f"    Confusion matrix:\n{cm}")
            pscore_train = sum(pscore_trains) / cfg.classification.cv_splits
            pscore_test = sum(pscore_tests) / cfg.classification.cv_splits

            pscore_train_avgs.append(pscore_train)
            pscore_test_avgs.append(pscore_test)
            tee(fout, f"Overall train accuracy for '{method}': {pscore_train}")
            tee(fout, f"Overall test accuracy for '{method}': {pscore_test}")
            tee(fout, "")

    df = pd.DataFrame({
        'pscore_train': pscore_train_avgs,
        'pscore_test': pscore_test_avgs,
        'model': cfg.classification.methods,
    })
    df['dataset'] = cfg.dataset.name
    sweep_vars = get_sweep_vars(outdir, cfg)
    for var, val in sweep_vars.items():
        df[var] = val
    df = df[['dataset'] + list(sweep_vars.keys()) + ['model', 'pscore_train', 'pscore_test']]
    df.to_csv(outdir / 'results.csv', index=False)

    print(f"Done with classification! Results at '{outdir}'")


def get_sweep_vars(outdir, cfg):
    '''
    Get relevant sweep variables and their values by parsing outdir
    '''
    sweep_vars = {}
    for subdir in outdir.parts:
        parts = subdir.split(',')
        if len(parts) == 1:
            continue
        stage = parts[0]
        for part in parts[1:]:
            var, value = part.split('=')
            try:
                value = int(value)
            except ValueError:
                pass
            sweep_vars[f'{stage}.{var}'] = value
    return sweep_vars



def tee(fout, text):
    print(text)
    fout.write(text)
    fout.write("\n")

def classification_task_rf(X_train, X_test, y_train, y_test, cfg, outdir):
    rf = Pipeline([
        ("rf", RandomForestClassifier(random_state=7, min_samples_split=5))
    ])
    parameters = {
        'rf__n_estimators': [10, 20, 50, 70, 100, 250],
    }
    greg = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, verbose=0)
    greg.fit(X_train, y_train)
    return report_metric(greg, X_train, X_test, y_train, y_test)


def classification_task_logr(X_train, X_test, y_train, y_test, cfg, outdir):
    lr = Pipeline([
        ("lr", LogisticRegression(random_state=7, penalty="elasticnet", solver="saga", max_iter=2000))
    ])
    parameters = {
        "lr__l1_ratio": [0.1, 0.3, 0.9, 1]
    }
    greg = GridSearchCV(estimator=lr, param_grid=parameters, cv=2, verbose=0)
    greg.fit(X_train, y_train)
    return report_metric(greg, X_train, X_test, y_train, y_test)


def classification_task_nn(X_train, X_test, y_train, y_test, cfg, outdir):
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
    return report_metric(model, X_train, X_test, y_train, y_test, argmax=True)



METHOD2FUNC = {
    'nn': classification_task_nn,
    'logistic': classification_task_logr,
    'random_forest': classification_task_rf,
}
