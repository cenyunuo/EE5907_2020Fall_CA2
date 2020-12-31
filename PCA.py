import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from common import Mean, Cent, ReducedPC

def PCA_vis(X_train, data_img, data_label, dimensions):
    if dimensions == 2:
        U_r = ReducedPC(X_train)
        vis = np.transpose(np.transpose(U_r)@Cent(data_img))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(vis[:, 0], vis[:, 1], c=data_label)
        legend1 = ax.legend(*scatter.legend_elements(num=25), loc=(1.1, 0), title="classes", ncol=2)
        plt.plot(vis[data_label==26][:, 0], vis[data_label==26][:, 1], 'rx', markersize=15)
    elif dimensions == 3:
        U_r = ReducedPC(X_train, dimensions = 3)
        vis = np.transpose(np.transpose(U_r)@Cent(data_img))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(vis[:, 0], vis[:, 1], vis[:, 2], c=data_label)
        legend1 = ax.legend(*scatter.legend_elements(num=25), loc=(1.1, 0), title="classes", ncol=2)
        plt.plot(vis[data_label==26][:, 0], vis[data_label==26][:, 1], vis[data_label==26][:, 2], 'rx', markersize=15)
    else:
        print('Can only visualize dimensions of 2 or 3')
    ax.add_artist(legend1)
    plt.show()

def PCA_face(X_train, y_train, dimensions):
    U_r = ReducedPC(X_train, dimensions = dimensions)
    img_re = (Mean(X_train) + U_r@np.transpose(U_r)@Cent(X_train)).reshape(32,32,len(y_train))
    fig, axs= plt.subplots(1, 3)
    for i in range(3):
    #Reconstructed faces
        axs[i].imshow(img_re[:, :, i], cmap='gray')
        axs[i].axis('off')
    plt.show()

def PCA_classify(X_train, y_train, X_test, y_test, dimensions):
    U_r = ReducedPC(X_train, dimensions = dimensions)
    X_train = np.transpose(np.transpose(U_r)@Cent(X_train))
    X_test = np.transpose(np.transpose(U_r)@Cent(X_test))
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    pred_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, pred_test)
    print('Dimensions: {}, [Training accuracy: {}, Validation accuracy: {}]'.format(dimensions, acc_train, acc_test))
    return acc_train, acc_test
