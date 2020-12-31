import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.mixture import GaussianMixture
from common import make_ellipses, Mean, Cent, ReducedPC

def GMM_vis(X, Prob, i):
    X = np.column_stack([X.reshape([1278,1024]), Prob[:,i].reshape(1278,1)])
    X = X[X[:,1024].argsort()][1273:1279,0:1024]
    img = X.reshape([5, 32, 32])
    fig, axs= plt.subplots(1, 5)
    for i in range(5):
        axs[i].imshow(img[i, :, :], cmap='gray')
        axs[i].axis('off')
    plt.show()

def GMM_train(X_train, X_test, dimensions):
    #Features depends on the dimensions
    U_r = ReducedPC(X_train, dimensions = dimensions)
    X_train = np.transpose(np.transpose(U_r)@Cent(X_train))
    X_test = np.transpose(np.transpose(U_r)@Cent(X_test))
    #Fit the model
    model = GaussianMixture(n_components=3)
    model.fit(X_train)
    Predict = model.predict(X_test)
    #Visulizezation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(X_train[:, 0], X_train[:, 1], s=0.8, c=y_train)
    # Plot the test data with crosses
    ax.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=Predict)
    make_ellipses(model, ax)
    return model.predict_proba(X_test)

def GMM(X_train, X_test, dimensions):
    print('Visualization of the Gaussian Model in the First two components')
    Prob = GMM_train(X_train, X_test, dimensions)
    print('Top 5 probable images from Gaussian Model 1')
    GMM_vis(X_test, Prob, 0)
    print('Top 5 probable images from Gaussian Model 2')
    GMM_vis(X_test, Prob, 1)
    print('Top 5 probable images from Gaussian Model 3')
    GMM_vis(X_test, Prob, 2)
