import os
import re
import csv
import math
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from qa_utils.io import batch_to_device
from qa_utils.misc import Logger


def read_args(working_dir):
    """Read the arguments that were saved during training.

    Arguments:
        working_dir {str} -- Working directory

    Returns:
        dict[str, str] -- A dict that maps arguments to their values
    """
    args = {}
    with open(os.path.join(working_dir, 'args.csv')) as fp:
        for arg, value in csv.reader(fp):
            args[arg] = value
    return args


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


def get_ranking_metrics(all_scores, all_labels, k):
    """Calculate MAP and MRR@k scores.

    Arguments:
        all_scores {list[list[float]]} -- The scores
        all_labels {list[list[int]]} -- The relevance labels
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
    for scores, labels in zip(all_scores, all_labels):
        rank_indices = np.asarray(scores).argsort()[::-1]
        gt_indices = set(list(np.where(np.asarray(labels) > 0)[0]))
        aps.append(_ap(rank_indices, gt_indices))
        rrs.append(_rr(rank_indices, gt_indices))
    return np.mean(aps), np.mean(rrs)


def evaluate(model, dataloader, k, device, has_multiple_inputs):
    """Evaluate the model on a testset.

    Arguments:
        model {torch.nn.Module} -- Classifier model
        dataloader {torch.utils.data.DataLoader} -- Testset DataLoader
        k {int} -- Calculate MRR@k
        device {torch.device} -- Device to evaluate on
        has_multiple_inputs {bool} -- Whether the input is a list of tensors

    Returns:
        dict[str, float] -- All computed metrics
    """
    result = defaultdict(lambda: ([], []))
    for batch in tqdm(dataloader):
        q_ids, inputs, labels = batch
        if has_multiple_inputs:
            inputs = batch_to_device(inputs, device)
            predictions = model(*inputs).cpu().detach()
        else:
            predictions = model(inputs.to(device)).cpu().detach()
        for q_id, prediction, label in zip(q_ids.numpy(), predictions.numpy(), labels.numpy()):
            result[q_id][0].append(prediction[0])
            result[q_id][1].append(label)

    metric_vals = {}

    all_scores, all_labels = [], []
    for q_id, (scores, labels) in result.items():
        all_scores.append(scores)
        all_labels.append(labels)
    map_, mrr = get_ranking_metrics(all_scores, all_labels, k)
    metric_vals['map'] = map_
    metric_vals['mrr'] = mrr

    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))

    y_true, y_pred = [], []
    for scores, labels in result.values():
        y_true.extend(labels)
        y_pred.extend([round(_sigmoid(x)) for x in scores])
    metric_vals['acc'] = accuracy_score(y_true, y_pred)
    return metric_vals


def evaluate_all(model, working_dir, dev_dl, test_dl, k, device, has_multiple_inputs=False,
                 dev_metric='mrr', test_metrics=['map', 'mrr'], interval=1):
    """Evaluate each checkpoint in the working directory against the devset. Afterwards, evaluate
    the checkpoint with the highest dev metric against the testset. The results are saved in a log
    file.

    Arguments:
        model {torch.nn.Module} -- The model to test
        working_dir {str} -- The working directory
        dev_dl {torch.utils.data.DataLoader} -- Dev dataloader
        test_dl {torch.utils.data.DataLoader} -- Test dataloader
        k {int} -- Compute MRR@k
        device {torch.device} -- Device to evaluate on

    Keyword Arguments:
        has_multiple_inputs {bool} -- Whether the input is a a list of tensors (default: {False})
        dev_metric {str} -- The metric to use for validation (default: {'mrr'})
        test_metrics {list[str]} -- The metrics to report on the testset (default: {['map', 'mrr']})
        interval {int} -- Evaluate only one in this many checkpoints (default: {1})
    """
    dev_file = os.path.join(working_dir, 'dev.csv')
    dev_logger = Logger(dev_file, ['ckpt', dev_metric])
    best = 0
    best_ckpt = None
    model.eval()
    for i, ckpt in enumerate(get_checkpoints(os.path.join(working_dir, 'ckpt'), r'weights_(\d+).pt')):
        if (i + 1) % interval != 0:
            continue

        print('[dev] processing {}...'.format(ckpt))
        state = torch.load(ckpt)
        model.module.load_state_dict(state['state_dict'])
        with torch.no_grad():
            metrics = evaluate(model, dev_dl, k, device, has_multiple_inputs)
        dev_logger.log([ckpt, metrics[dev_metric]])
        if metrics[dev_metric] >= best:
            best = metrics[dev_metric]
            best_ckpt = ckpt

    print('[test] processing {}...'.format(best_ckpt))
    state = torch.load(best_ckpt)
    model.module.load_state_dict(state['state_dict'])
    with torch.no_grad():
        metrics = evaluate(model, test_dl, k, device, has_multiple_inputs)

    test_file = os.path.join(working_dir, 'test.csv')
    test_logger = Logger(test_file, ['ckpt'] + test_metrics)
    test_logger.log([best_ckpt] + [metrics[m] for m in test_metrics])
