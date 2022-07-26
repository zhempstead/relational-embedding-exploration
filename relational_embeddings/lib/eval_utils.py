import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def show_stats(model, X_train, X_test, y_train, y_test, argmax=False, metric=accuracy_score, fout=sys.stdout):
    X_pred_train = model.predict(X_train)
    X_pred_test = model.predict(X_test)
    if argmax == True:
        X_pred_train = np.argmax(X_pred_train, axis=1)
        X_pred_test = np.argmax(X_pred_test, axis=1)
    fout.write(f"an pred: {y_test[:30]}")
    fout.write(f"my pred: {X_pred_test[:30]}")
    pscore_train = metric(y_train, X_pred_train)
    pscore_test = metric(y_test, X_pred_test)
    # print("Confusion Matrix:", confusion_matrix(y_test, X_pred_test))
    fout.write(f"Train accuracy {pscore_train}, Test accuracy {pscore_test}")
    return pscore_train, pscore_test


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
