import os
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def get_checkpoints(directory, pattern):
    """List all model checkpoints.
    
    Arguments:
        directory {str} -- Directory that contains checkpoint files
        pattern {str} -- Regex pattern to match the files
    
    Returns:
        list[str] -- A sorted list of absolute paths to all checkpoint files
    """
    files = filter(lambda x: re.match(pattern, x), os.listdir(directory))
    files = map(lambda x: os.path.join(os.path.abspath(directory), x), files)
    files = filter(os.path.isfile, files)
    return sorted(files)


def get_metrics(scores_list, labels_list, k):
    """Calculate MAP and MRR@k scores.
    
    Arguments:
        scores_list {list[list[float]]} -- The scores
        labels_list {list[list[int]]} -- The relevance labels
        k {int} -- Calculate MRR@k
    
    Returns:
        tuple[float, float] -- A tuple containing MAP and MRR@k
    """
    def _ap(pred, gt):
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(pred):
            if p in gt and p not in pred[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / max(1.0, len(gt))

    def _rr(pred, gt):
        score = 0.0
        for rank, item in enumerate(pred[:k]):
            if item in gt:
                score = 1.0 / (rank + 1.0)
                break
        return score

    aps, rrs = [], []
    for scores, labels in zip(scores_list, labels_list):
        rank_indices = np.asarray(scores).argsort()[::-1]
        gt_indices = set(list(np.where(np.asarray(labels) > 0)[0]))
        aps.append(_ap(rank_indices, gt_indices))
        rrs.append(_rr(rank_indices, gt_indices))
    return np.mean(aps), np.mean(rrs)


def evaluate(model, dataloader, k):
    """Evaluate the model on a testset.
    
    Arguments:
        model {torch.nn.Module} -- Classifier model
        dataloader {torch.utils.data.DataLoader} -- Testset DataLoader
        k {int} -- Calculate MRR@k
    
    Returns:
        tuple[float, float] -- A tuple containing MAP and MRR@k
    """
    result = defaultdict(lambda: ([], []))
    for batch in tqdm(dataloader):
        q_ids, inputs, labels = batch
        predictions = model(inputs).cpu().detach()
        for q_id, prediction, label in zip(q_ids.numpy(), predictions.numpy(), labels.numpy()):
            result[q_id][0].append(prediction[0])
            result[q_id][1].append(label)

    all_scores, all_labels = [], []
    for q_id, (score, label) in result.items():
        all_scores.append(score)
        all_labels.append(label)
    map_, mrr = get_metrics(all_scores, all_labels, k)
    return map_, mrr
