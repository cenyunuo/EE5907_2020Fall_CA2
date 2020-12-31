import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from common import Mean, Cent, ReducedPC

def LDA_vis(X_train, y_train, data_img, data_label, dimensions):
    X_train = np.transpose(Cent(X_train))
    data_img = np.transpose(Cent(data_img))
    if dimensions == 2:
        model = LinearDiscriminantAnalysis(n_components=2)
        model.fit(X_train, y_train)
        X_transformed = model.transform(data_img)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=data_label)
        legend1 = ax.legend(*scatter.legend_elements(num=25), loc=(1.1, 0), title="classes", ncol=2)
        plt.plot(X_transformed[data_label==26][:, 0], X_transformed[data_label==26][:, 1], 'rx', markersize=15)
    elif dimensions == 3:
        model = LinearDiscriminantAnalysis(n_components=3)
        model.fit(X_train, y_train)
        X_transformed = model.transform(data_img)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=data_label)
        legend1 = ax.legend(*scatter.legend_elements(num=25), loc=(1.1, 0), title="classes", ncol=2)
        plt.plot(X_transformed[data_label==26][:, 0], X_transformed[data_label==26][:, 1], X_transformed[data_label==26][:, 2], 'r*', markersize=20)
    else:
        print('Can only visualize dimensions of 2 or 3')
    ax.add_artist(legend1)
    plt.show()

def LDA_classify(X_train, y_train, X_test, y_test, dimensions):
    X_train = np.transpose(Cent(X_train))
    X_test = np.transpose(Cent(X_test))
    model_lda = LinearDiscriminantAnalysis(n_components=dimensions)
    model_lda.fit(X_train, y_train)
    X_train = model_lda.transform(X_train)
    X_test = model_lda.transform(X_test)
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train, y_train)
    pred_train = model_knn.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    pred_test = model_knn.predict(X_test)
    acc_test = accuracy_score(y_test, pred_test)
    print('Dimensions: {}, [Training accuracy: {}, Validation accuracy: {}]'.format(dimensions, acc_train, acc_test))
    return acc_train, acc_test
