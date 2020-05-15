import numpy as np


def ap(predictions, labels):
    """Calculate the average precision for a single query.

    Arguments:
        predictions {list[float]} -- A list of predictions
        labels {list[int]} -- A list of labels

    Returns:
        float -- The average precision
    """
    prediction_indices = np.asarray(predictions).argsort()[::-1]
    label_indices = set(list(np.where(np.asarray(labels) > 0)[0]))
    score = 0
    num_hits = 0
    for i, p in enumerate(prediction_indices):
        if p in label_indices and p not in prediction_indices[:i]:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / max(1, len(label_indices))


def rr(predictions, labels, k):
    """Calculate the reciprocal rank for a single query.

    Arguments:
        predictions {list[float]} -- A list of predictions
        labels {list[int]} -- A list of labels
        k {int} -- Calculate RR@k

    Returns:
        float -- The reciprocal rank
    """
    prediction_indices = np.asarray(predictions).argsort()[::-1]
    label_indices = set(list(np.where(np.asarray(labels) > 0)[0]))
    for rank, item in enumerate(prediction_indices[:k]):
        if item in label_indices:
            return 1 / (rank + 1)
    return 0
