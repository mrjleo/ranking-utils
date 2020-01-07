import os
import re
import csv
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from qa_utils.misc import Logger


def read_args(working_dir):
    """Read the arguments that were saved during training.

    Arguments:
        working_dir {str} -- Working directory

    Returns:
        dict -- A dict that maps arguments to their values
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


def evaluate(model, dataloader, k, device):
    """Evaluate the model on a testset.

    Arguments:
        model {torch.nn.Module} -- Classifier model
        dataloader {torch.utils.data.DataLoader} -- Testset DataLoader
        k {int} -- Calculate MRR@k
        device {torch.device} -- Device to evaluate on

    Returns:
        tuple[float, float] -- A tuple containing MAP and MRR@k
    """
    result = defaultdict(lambda: ([], []))
    for batch in tqdm(dataloader):
        q_ids, inputs, labels = batch
        predictions = model(inputs.to(device)).cpu().detach()
        for q_id, prediction, label in zip(q_ids.numpy(), predictions.numpy(), labels.numpy()):
            result[q_id][0].append(prediction[0])
            result[q_id][1].append(label)

    all_scores, all_labels = [], []
    for q_id, (score, label) in result.items():
        all_scores.append(score)
        all_labels.append(label)
    map_, mrr = get_metrics(all_scores, all_labels, k)
    return map_, mrr


def evaluate_all(model, working_dir, dev_dl, test_dl, k, device):
    """Evaluate each checkpoint in the working directory agains dev- and testset. Save the results in a log file.

    Arguments:
        model {torch.nn.Module} -- The model to test
        working_dir {str} -- The working directory
        dev_dl {torch.utils.data.DataLoader} -- Dev dataloader
        test_dl {torch.utils.data.DataLoader} -- Test dataloader
        k {int} -- Compute MRR@k
        device {torch.device} -- Device to evaluate on
    """
    eval_file = os.path.join(working_dir, 'eval.csv')
    logger = Logger(eval_file, ['ckpt', 'dev_map', 'dev_mrr', 'test_map', 'test_mrr'])
    model.eval()
    for ckpt in get_checkpoints(os.path.join(working_dir, 'ckpt'), r'weights_(\d+).pt'):
        print('processing {}...'.format(ckpt))
        state = torch.load(ckpt)
        model.module.load_state_dict(state['state_dict'])
        with torch.no_grad():
            dev_metrics = evaluate(model, dev_dl, k, device)
            test_metrics = evaluate(model, test_dl, k, device)
        row = [ckpt] + list(dev_metrics) + list(test_metrics)
        logger.log(row)
