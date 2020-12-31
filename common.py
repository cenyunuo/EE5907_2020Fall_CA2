import os
import numpy as np
from numpy.matlib import repmat
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib as mpl

def dataloader(dir, number = 170, split = True, gmm = False):
    files = [file for file in os.listdir(dir)]
    data_img = []
    data_label = []
    for file in files:
        wd = os.path.join(dir, file)
        i = 0
        for img_file in os.listdir(wd):
            i = i+1
            path = os.path.join(wd, img_file)
            img = plt.imread(path)
            data_img.append(img)
            data_label.append(int(file))
            if i == number:
                break
    data_img = np.array(data_img)
    data_label = np.array(data_label)
    if split == False:
        return data_img, data_label
    else:
        return train_test_split(data_img,data_label,test_size=0.3, random_state=0)

def Mean(X):
    #Reshape the image from (32, 32) to 1024
    X = np.transpose(X.reshape(X.shape[0],1024))
    return repmat(np.mean(X, axis = 1).reshape(1024,1),1, X.shape[1])

def Cent(X):
    #Centerized the data points
    A = np.transpose(X.reshape(X.shape[0],1024))
    return A - Mean(X)

def ReducedPC(X, dimensions = 2):
    #SVD with normalized value
    U, S, V = np.linalg.svd(Cent(X))
    #Reduced principal component
    U_r = U[:,0:dimensions]
    return U_r

def make_ellipses(gmm, ax):
    colors = ['navy', 'turquoise', 'darkorange']
    for n in range(3):
        covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color = colors[n])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')
