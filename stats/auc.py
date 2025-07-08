import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def compute_auc_ci(y_true, y_prob, n_bootstraps=1000, ci=0.95):
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int((1.0 - ci) / 2.0 * len(sorted_scores))]
    upper = sorted_scores[int((1.0 + ci) / 2.0 * len(sorted_scores))]
    return np.mean(bootstrapped_scores), lower, upper