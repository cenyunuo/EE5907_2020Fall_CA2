import numpy as np
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score
from common import Mean, Cent, ReducedPC


def SVM(X_train, y_train, X_test, y_test, dimensions = 80, penalty = 1):
    if dimensions == 1024:
        X_train = preprocessing.scale(X_train.reshape(len(y_train),1024))
        X_test = preprocessing.scale(X_test.reshape(len(y_train),1024))
    else:
    # PCA data preprocessing
        U_r = ReducedPC(X_train, dimensions = dimensions)
        X_train = preprocessing.scale(np.transpose(np.transpose(U_r)@Cent(X_train)))
        X_test = preprocessing.scale(np.transpose(np.transpose(U_r)@Cent(X_test)))
    # Fitting model
    model = svm.LinearSVC(C = penalty, tol = 0.1, max_iter = 2000)
    model.fit(X_train, y_train)
    # Evaluation
    pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    pred_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, pred_test)
    print('Dimensions: {}, Penalty: {} [Training accuracy: {}, Validation accuracy: {}]'.format(dimensions, penalty, acc_train, acc_test))
