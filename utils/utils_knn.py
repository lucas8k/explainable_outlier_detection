import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def fit_predict_knn(X, n_neighbors=30, algorithm="brute"):
    neigh = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm=algorithm)
    neigh.fit(X)
    # calculate the n nearest neighbors for the given dataset
    knn_scores, neighbors = neigh.kneighbors(X, n_neighbors + 1, return_distance=True)
    # calculate the global knn gy caluclating the avg and exluding the first one
    scores = [sum(scores) / len(scores - 1) for scores in knn_scores]
    return scores, neighbors


def get_auc(df):
    fpr, tpr, thresholds = metrics.roc_curve(df['label'], df['score'], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def gridsearch_n(X, y, min_n=1, max_n=30, algorithm="brute", plot=True):
    aucs = {}
    for n in range(min_n, max_n + 1):
        scores, _ = fit_predict_knn(X, n, algorithm)
        scored_df = pd.DataFrame(X)
        scored_df.reset_index()
        scored_df["score"] = scores
        scored_df["label"] = y
        aucs[n] = get_auc(scored_df)
    if plot:
        plt.figure(figsize=[8, 5])
        plt.plot(list(aucs.keys()), list(aucs.values()))
        plt.ylabel('AUC')
        plt.xlabel('Number of Neighbors')
        best_auc_idx = np.argsort(list(aucs.values()))[-1]
        plt.title(
            'Best AUC{} at {} neighbors'.format(list(aucs.values())[best_auc_idx], list(aucs.keys())[best_auc_idx]))
    return aucs


def plot_roc_curve(df):
    fpr, tpr, thresholds = metrics.roc_curve(df['label'], df['score'], pos_label=1)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=[8, 5])
    plt.plot(fpr, tpr, color='r', lw=2, label='Global KNN')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: AUC = {0:0.4f}'.format(auc))
    plt.legend(loc="lower right")
    plt.show()


def dist_per_dimension(X, neighbors, idx):
    dist = np.zeros((len(idx), X.shape[1]))
    for j, i in enumerate(idx):
        dist[j] = np.abs((X[neighbors[:, 1:][i]] - X[i]).mean(axis=0))
    return dist

def scale_data(X):
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    return X_scaled

