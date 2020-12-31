import os
import argparse
import matplotlib.pyplot as plt


from common import dataloader, make_ellipses, Mean, Cent, ReducedPC
from PCA import PCA_vis, PCA_face, PCA_classify
from LDA import LDA_vis, LDA_classify
from GMM import GMM
from SVM import SVM
from CNN import CNN


parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('-a', '--algorithm', type=str,
                default=None, help='Algorithm')
parser.add_argument('-m', '--method', type=str,
                default=None, help='method')
parser.add_argument('-d' ,'--dimensions', type=int,
                default=None, help='Dimension')
parser.add_argument('-p' ,'--parameters', type=float,
                default=None, help='Parameters')

args = parser.parse_args()

def main(args):
    X_train, X_test, y_train, y_test = dataloader('PIE')

    data_img, data_label = dataloader('PIE', number = 20, split = False)

    if args.algorithm == 'PCA':
        if args.method == 'vis':
            PCA_vis(X_train, data_img, data_label, args.dimensions)
        elif args.method == 'face':
            PCA_face(X_train, y_train, args.dimensions)
        elif args.method == 'classify':
            acc_train, acc_test = PCA_classify(X_train, y_train, X_test, y_test, args.dimensions)
        else:
            print('command not found')

    elif args.algorithm == 'LDA':
        if args.method == 'vis':
            LDA_vis(X_train, y_train, data_img, data_label, args.dimensions)
        elif args.method == 'classify':
            LDA_classify(X_train, y_train, X_test, y_test, args.dimensions)
        else:
            print('command not found')

    elif args.algorithm == 'GMM':
        if args.method == 'clustering':
            GMM(X_train, X_test, args.dimensions)
        else:
            print('command not found')

    elif args.algorithm == 'SVM':
        if args.method == 'classify':
            SVM(X_train, y_train, X_test, y_test, args.dimensions, args.parameters)
        else:
            print('command not found')

    elif args.algorithm == 'CNN':
        if args.method == 'classify':
            model, history = CNN(X_train, y_train, X_test, y_test, args.parameters)
            model.summary()
            plt.plot(history.history['accuracy'], label='train acc')
            plt.plot(history.history['val_accuracy'], label = 'test acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            plt.legend(loc='lower right')
            plt.show()
        else:
            print('command not found')

    else:
        print('command not found')

main(args)
