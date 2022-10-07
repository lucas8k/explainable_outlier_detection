import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

def fit_predict_gm(X, n_components=5, SEED=42):
    gm = GaussianMixture(n_components=n_components, random_state=SEED).fit(X)
    scores = gm.score_samples(X) * -1
    return scores, gm

def get_auc(df):
    fpr, tpr, thresholds = metrics.roc_curve(df['label'], df['score'], pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc

def gridsearch_n(X, y, min_n=1, max_n=30, plot=True, SEED=42):
    aucs = {}
    for n in range(min_n, max_n+1): 
        scores, _ = fit_predict_gm(X, n, SEED=SEED)
        scored_df = pd.DataFrame(X)
        scored_df.reset_index()
        scored_df["score"] = scores
        scored_df["label"] = y
        aucs[n] = get_auc(scored_df)
    if plot:
        plt.figure(figsize=[8,5])
        plt.plot(list(aucs.keys()), list(aucs.values()))
        plt.ylabel('AUC')
        plt.xlabel('Number of Components')
        best_auc_idx = np.argsort(list(aucs.values()))[-1]
        plt.title('Best AUC{} at {} components'.format(list(aucs.values())[best_auc_idx], list(aucs.keys())[best_auc_idx]))
    return aucs


def plot_roc_curve(df):
    fpr, tpr, thresholds = metrics.roc_curve(df['label'], df['score'], pos_label=1)
    auc = metrics.auc(fpr,tpr)

    plt.figure(figsize=[8,5])
    plt.plot(fpr, tpr, color='r', lw=2, label='GMM')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: AUC = {0:0.4f}'.format(auc))
    plt.legend(loc="lower right")
    plt.show()  


def cov_to_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def mah_dist_per_dimension(means, covs, l, X):
    """calculates the mahalanobis distance per dimension"""
    X = X.astype("float")
    cov = covs[l].astype("float")
    mean = means[l].astype("float")
    cov_inv = np.linalg.inv(cov)
    x_m = X - mean
    tmp = np.matmul(x_m, cov_inv)
    mah_d = np.multiply(tmp, x_m.T)
    return np.sqrt(np.abs(mah_d))

def mah_dist_per_dim(gm, X, I):
    mask_o = gm.weights_.round(3) > 0.1
    dist = np.zeros((len(I), X.shape[1]))
    cs = np.zeros(len(I))
    for idx, i in enumerate(I):
        #c = gm.predict(X[i].reshape(1,-1))[0]
        proba = gm.predict_proba(X[i].reshape(1,-1))[0]
        sort_idx = np.argsort(proba)[::-1]
        mask = mask_o[sort_idx]
        c = sort_idx[mask][0]
        cs[idx] = c
        dist[idx] = mah_dist_per_dimension(gm.means_, gm.covariances_, c, X[i])
    return dist, cs[I]


def draw_ellipse(position, covariance, ax=None, nr_patches=4, c="b", label=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 5 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    
    # Draw the Ellipse
    label_set = False
    for nsig in range(1, nr_patches):
        if label is not None and not label_set:
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, color=c, label=label, **kwargs))
            label_set = True
        else:
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, color=c, **kwargs))         

def draw_components(gm, columns, ax=None, **kwargs):
    colors = ["b", "r", "g", "c", "m", "y", "k", "x"]
    w_factor = 0.5 / gm.weights_.max()
    for idx, (pos, covar, w) in enumerate(zip(gm.means_[:, columns], gm.covariances_[:, columns, :][:, :, columns], gm.weights_)):
        draw_ellipse(pos, covar, alpha=0.3, ax=ax, c=colors[idx])

