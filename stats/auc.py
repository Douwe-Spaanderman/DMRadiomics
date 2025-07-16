import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

from typing import Tuple


def compute_auc_ci(
    y_true, y_prob, n_bootstraps=1000, ci=0.95
) -> Tuple[float, float, float]:
    """
    Compute the AUC and its confidence interval using bootstrapping.
    Parameters:
    - y_true: array-like, true binary labels (0 or 1).
    - y_prob: array-like, predicted probabilities for the positive class.
    - n_bootstraps: int, number of bootstrap samples to draw.
    - ci: float, confidence level for the interval (default is 0.95).
    Returns:
    - mean_auc: float, mean AUC from bootstrapping.
    - lower_bound: float, lower bound of the confidence interval.
    - upper_bound: float, upper bound of the confidence interval.
    """
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
