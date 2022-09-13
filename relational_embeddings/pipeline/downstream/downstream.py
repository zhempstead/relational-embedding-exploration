from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from relational_embeddings.lib.eval_utils import plot_tf_history, report_metric
from relational_embeddings.lib.utils import dataset_dir, get_sweep_vars

def downstream(cfg, outdir, indir=None):
    """
    Evaluate model on downstream ML task (as determined by dataset)
    """
    if indir is None:
        indir = outdir.parent

    task = cfg.dataset.downstream_task
    random_state = np.random.RandomState(seed=cfg.downstream.random_seed)

    df_x = pd.read_csv(indir / 'embeddings.csv')
    df_y = pd.read_csv(dataset_dir(cfg.dataset.name) / 'base_y.csv')

    shuffled_idx = random_state.permutation(len(df_x))
    df_x = df_x.reindex(shuffled_idx)
    df_y = df_y.reindex(shuffled_idx)
    if task == 'classification':
        df_y[cfg.dataset.target_column] = pd.Categorical(df_y[cfg.dataset.target_column]).codes
        kfold_class = StratifiedKFold
    elif task == 'regression':
        kfold_class = KFold

    kfold = kfold_class(n_splits=cfg.downstream.cv_splits)
    train_test_idxs = list(kfold.split(df_x, df_y))

    pscore_train_avgs = []
    pscore_test_avgs = []

    with open(outdir / 'results.txt', 'w') as fout:
        for method in cfg.downstream.methods_by_task[task]:
            tee(fout, f"{task} method '{method}':")
            method_func = TASK2METHOD2FUNC[task][method]
            pscore_trains = []
            pscore_tests = []
            for it in range(cfg.downstream.cv_splits):
                tee(fout, f"  Iteration {it+1}/{cfg.downstream.cv_splits}:")
                train_idx, test_idx = train_test_idxs[it]
                df_train_x = df_x.iloc[train_idx]
                df_test_x = df_x.iloc[test_idx]
                df_train_y = df_y.iloc[train_idx]
                df_test_y = df_y.iloc[test_idx]
                pscore_train, pscore_test, cm, y_sample, pred_sample = \
                    method_func(df_train_x, df_test_x, df_train_y, df_test_y, cfg.downstream, outdir)
                pscore_trains.append(pscore_train)
                pscore_tests.append(pscore_test)
                tee(fout, f"    real: {y_sample}")
                tee(fout, f"    pred: {pred_sample}")
                tee(fout, f"    Train accuracy {pscore_train}, Test accuracy {pscore_test}")
                tee(fout, f"    Confusion matrix:\n{cm}")
            pscore_train = sum(pscore_trains) / cfg.downstream.cv_splits
            pscore_test = sum(pscore_tests) / cfg.downstream.cv_splits

            pscore_train_avgs.append(pscore_train)
            pscore_test_avgs.append(pscore_test)
            tee(fout, f"Overall train accuracy for '{method}': {pscore_train}")
            tee(fout, f"Overall test accuracy for '{method}': {pscore_test}")
            tee(fout, "")

    df = pd.DataFrame({
        'pscore_train': pscore_train_avgs,
        'pscore_test': pscore_test_avgs,
        'model': cfg.downstream.methods_by_task[task],
    })
    df['dataset'] = cfg.dataset.name
    sweep_vars = get_sweep_vars(outdir)
    for var, val in sweep_vars.items():
        df[var] = val
    df = df[['dataset'] + list(sweep_vars.keys()) + ['model', 'pscore_train', 'pscore_test']]
    df.to_csv(outdir / 'results.csv', index=False)

    print(f"Done with {task}! Results at '{outdir}'")


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
    greg.fit(X_train, y_train[y_train.columns[0]])
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


def regression_task_elasticnet(X_train, X_test, y_train, y_test, cfg, outdir):
    en = Pipeline([
        ("normalizer", Normalizer()),
        ("en", ElasticNet(normalize=True, random_state=7, max_iter=100))
    ])
    parameters = {
        'en__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
        'en__l1_ratio': [0.2, 0.5, 0.8]
    }
    greg = GridSearchCV(estimator=en, param_grid=parameters, cv=5, verbose=0)
    greg.fit(X_train, y_train)
    return report_metric(greg, X_train, X_test, y_train, y_test, metric=r2_score)


def regression_task_nn(X_train, X_test, y_train, y_test, cfg, outdir):
    input_size = X_train.shape[1]
    def baseline_model():
        # create model
        model = Sequential()
        model.add(tf.keras.layers.Dense(input_size, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(input_size // 2, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(input_size // 4, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mae', optimizer='adam')
        return model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=10, verbose=1)))
    pipeline = Pipeline(estimators)
    pipeline.fit(X_train, y_train)
    return report_metric(pipeline, X_train, X_test, y_train, y_test, metric=r2_score)


TASK2METHOD2FUNC = {
    'classification': {
        'nn': classification_task_nn,
        'logistic': classification_task_logr,
        'random_forest': classification_task_rf,
    },
    'regression': {
        'nn': regression_task_nn,
        'elastic': regression_task_elasticnet
    },
}
