import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from relational_embeddings.lib.utils import tee

def train_downstream_model(df_x, df_y, cfg, outdir, teefile):
    """
    For classification and regression tasks, train a model over 5 KFold splits and take the average
    """
    tf.keras.utils.set_random_seed(cfg.random_seed)

    shuffled_idx = np.random.permutation(len(df_x))
    df_x = df_x.reindex(shuffled_idx)
    df_y = df_y.reindex(shuffled_idx)
    if cfg.task == 'classification':
        target_column = df_y.columns[0]
        df_y[target_column] = pd.Categorical(df_y[target_column]).codes
        kfold_class = StratifiedKFold
    elif cfg.task == 'regression':
        kfold_class = KFold

    kfold = kfold_class(n_splits=cfg.cv_splits)
    train_test_idxs = list(kfold.split(df_x, df_y))

    pscore_train_avgs = []
    pscore_test_avgs = []

    with open(teefile, 'w') as fout:
        for method in cfg.methods_by_task[cfg.task]:
            tee(fout, f"{cfg.task} method '{method}':")
            method_func = TASK2METHOD2FUNC[cfg.task][method]
            pscore_trains = []
            pscore_tests = []
            for it in range(cfg.cv_splits):
                tee(fout, f"  Iteration {it+1}/{cfg.cv_splits}:")
                train_idx, test_idx = train_test_idxs[it]
                df_train_x = df_x.iloc[train_idx]
                df_test_x = df_x.iloc[test_idx]
                df_train_y = df_y.iloc[train_idx]
                df_test_y = df_y.iloc[test_idx]
                pscore_train, pscore_test, cm, y_sample, pred_sample = \
                    method_func(df_train_x, df_test_x, df_train_y, df_test_y, cfg, outdir)
                pscore_trains.append(pscore_train)
                pscore_tests.append(pscore_test)
                tee(fout, f"    real: {y_sample}")
                tee(fout, f"    pred: {pred_sample}")
                tee(fout, f"    Train accuracy {pscore_train}, Test accuracy {pscore_test}")
                tee(fout, f"    Confusion matrix:\n{cm}")
            pscore_train = sum(pscore_trains) / cfg.cv_splits
            pscore_test = sum(pscore_tests) / cfg.cv_splits

            pscore_train_avgs.append(pscore_train)
            pscore_test_avgs.append(pscore_test)
            tee(fout, f"Overall train accuracy for '{method}': {pscore_train}")
            tee(fout, f"Overall test accuracy for '{method}': {pscore_test}")
            tee(fout, "")

    return pd.DataFrame({
        'model': cfg.methods_by_task[cfg.task],
        'pscore_train': pscore_train_avgs,
        'pscore_test': pscore_test_avgs,
    })


def classification_task_rf(X_train, X_test, y_train, y_test, cfg, outdir):
    rf = Pipeline([
        ("rf", RandomForestClassifier(min_samples_split=5))
    ])
    parameters = {
        'rf__n_estimators': [10, 20, 50, 70, 100, 250],
    }
    greg = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, verbose=0)
    greg.fit(X_train, y_train)
    return report_metric(greg, X_train, X_test, y_train, y_test)


def classification_task_logr(X_train, X_test, y_train, y_test, cfg, outdir):
    lr = Pipeline([
        ("lr", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=2000))
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
        ("standardize", StandardScaler()),
        ("en", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1]))
    ])
    en.fit(X_train, y_train)
    return report_metric(en, X_train, X_test, y_train, y_test, metric=mean_absolute_error)


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
    return report_metric(pipeline, X_train, X_test, y_train, y_test, metric=mean_absolute_error)


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
def report_metric(model, X_train, X_test, y_train, y_test, argmax=False, metric=accuracy_score):
    X_pred_train = model.predict(X_train)
    X_pred_test = model.predict(X_test)
    if argmax:
        X_pred_train = np.argmax(X_pred_train, axis=1)
        X_pred_test = np.argmax(X_pred_test, axis=1)
    y_sample = y_test[:30]
    pred_sample = X_pred_test[:30]
    pscore_train = metric(y_train, X_pred_train)
    pscore_test = metric(y_test, X_pred_test)
    try:
      cm = confusion_matrix(y_test, X_pred_test)
    except ValueError:
      cm = None
    return pscore_train, pscore_test, cm, y_sample, pred_sample


def plot_tf_history(history, outfile=None):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if outfile is None:
        plt.show()
    else:
        plt.savefig(str(outfile) + '_acc.png')

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if outfile is None:
        plt.show()
    else:
        plt.savefig(str(outfile) + '_history.png')

    plt.clf()
