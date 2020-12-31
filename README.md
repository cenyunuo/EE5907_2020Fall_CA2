# EE5907_2020Fall_CA2
### Introduction
This is the readme file for desrbibing the codes for EE5907 CA2.

In this course project, we eavluate several algorithms: PCA/LDA/GMM/SVM/CNN, on CMU PIE dataset.

Here is the content of this project. Please do arrange your documents in such way:
```
├── Readme.md                   
├── PIE
│   ├── 1
│   ├── 2
│   ├── ...
│   └── 26
├── main.py
└── common.py    
```

### Preliminaries

Running this code requires:
1. [numpy](https://numpy.org/)
2. [tensorflow](https://www.tensorflow.org/)
3. [matplotlib](https://matplotlib.org/)
4. [sci-kit learn](https://scikit-learn.org/stable/)
5. `common.py` (Some common functions required in `main.py`, but seperate to another document to make code looks neat).
6. Make sure you include the dataset in the folder `PIE` in the same path. And make sure the total class of the dataset should be 26.

### Usage

This code can support 6 algorithm for evaluate the `PIE` dataset. You can run the codes like this:

```
python main.py -a ... -m ... -d ... -p ...
```
Since we have totally 5 algorithms here, and they have different methods and parameters, so we should specify them when we run the codes.

- `-a`: Algorithm, where you should specify whether is **PCA/LDA/GMM/SVM/CNN**;
- `-m`: Method, where you should specity whether is what method you want to evaluate on the algorithm;
- `-d`: Dimension, where you should specify how many reduced dimensions you want to retain (`-d` is not required for CNN);
- `-p`: The parameters you want to apply (`-p` is not required for PCA, LDA and GMM).


- methods of `PCA`:
1. `vis`: Visualize the PCA result in 2 or 3 dimensions;
2. `face`: Reconstruct faces from reduced PCs;
3. `classify`: Classify with KNN classifiers with reduced PCs.

- methods of `LDA`:
1. `vis`: Visualize the LDA result in 2 or 3 dimensions;
2. `classify`: Classify with KNN classifiers with reduced PCs.

- methods of `GMM`:
1. `clustering`: Visualize the clustering result of GMM with reduced PCs;

- method of `SVM`:
1. `classify`: Classify with SVM classifiers with reduced PCs. And you should specify `-p` with penalty you want to apply.

- method of `CNN`:
1. `classify`: Classify with CNN model. And you should specify `-p` with epochs you want to train the model.

### Example

For exmaple, you want to visulize the PCA result in 2 dimensions:

```
python main.py -a PCA -m vis -d 2
```

Or, you want to train a SVM model with 40 reduced PCs and  penalty of 0.1:

```
python main.py -a SVM -m classify -d 40 -p 0.1
```

Or, you want to train the CNN with 20 epochs:

```
python main.py -a CNN -m classify -p 20
```
