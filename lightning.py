from collections import defaultdict

import torch
import numpy as np
from pytorch_lightning import LightningModule


def ap(predictions, labels):
    """Calculate the average precision for a single query.

    Arguments:
        predictions {torch.Tensor} -- A list of predictions
        labels {torch.Tensor} -- A list of labels

    Returns:
        float -- The average precision
    """
    prediction_indices = predictions.argsort(descending=True)
    label_indices, = torch.where(labels > 0)
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
        predictions {torch.Tensor} -- A list of predictions
        labels {torch.Tensor} -- A list of labels
        k {int} -- Compute RR@k

    Returns:
        float -- The reciprocal rank
    """
    prediction_indices = predictions.argsort(descending=True)
    label_indices, = torch.where(labels > 0)
    for rank, item in enumerate(prediction_indices[:k]):
        if item in label_indices:
            return 1 / (rank + 1)
    return 0


class BaseRanker(LightningModule):
    """Base class for re-rankers. Implements computation of and validation based on MAP and MRR.
    This class needs to be extended and (at least) the following methods must be implemented:
        * forward
        * configure_optimizers
        * training_step

    Arguments:
        train_ds {str} -- Training dataset
        val_ds {str} -- Validation dataset
        test_ds {str} -- Test dataset
        batch_size {int} -- Training/validation/test batch size
        mrr_k {int} -- Compute MRR@k
        num_workers {int} -- Number of DataLoader workers
    """
    def __init__(self, train_ds, val_ds, test_ds, batch_size, mrr_k=10, num_workers=16):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.mrr_k = mrr_k
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self):
        """Return a trainset DataLoader.

        Returns:
            torch.utils.data.DataLoader: The DataLoader
        """
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)
    
    def val_dataloader(self):
        """Return a validation set DataLoader.

        Returns:
            torch.utils.data.DataLoader: The DataLoader
        """
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)

    def validation_step(self, batch, _batch_idx):
        """Process a single validation batch.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Query IDs, inputs and labels
            _batch_idx (int): Batch index

        Returns:
            dict[str, torch.Tensor]: Query IDs, resulting predictions and labels
        """
        q_ids, inputs, labels = batch
        outputs = self(inputs)
        return {'q_ids': q_ids, 'predictions': outputs, 'labels': labels}

    def validation_epoch_end(self, results):
        """Accumulate all validation batches.

        Args:
            results (dict[str, torch.Tensor]): Query IDs, resulting predictions and labels

        Returns:
            dict[str, dict[str, torch.Tensor]]: Logged metrics (MAP and MRR@k)
        """
        r = defaultdict(lambda: ([], []))
        for step in results:
            for q_id, (pred,), label in zip(step['q_ids'], step['predictions'], step['labels']):
                # q_id is a tensor with one element, we convert it to an int to use it as dict key
                q_id = int(q_id.cpu())
                r[q_id][0].append(pred)
                r[q_id][1].append(label)
        aps, rrs = [], []
        for predictions, labels in r.values():
            predictions = torch.stack(predictions)
            labels = torch.stack(labels)
            aps.append(ap(predictions, labels))
            rrs.append(rr(predictions, labels, self.mrr_k))
        return {'log': {'map': torch.as_tensor(np.mean(aps)),
                        'mrr': torch.as_tensor(np.mean(rrs))}}
    
    def test_dataloader(self):
        """Return a testset DataLoader.

        Returns:
            torch.utils.data.DataLoader: The DataLoader
        """
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size,
                                           shuffle=False, num_workers=16)
    
    def test_step(self, batch, _batch_idx):
        """Process a single test batch.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Query IDs, inputs and labels
            _batch_idx (int): Batch index

        Returns:
            dict[str, torch.Tensor]: Query IDs, resulting predictions and labels
        """
        return self.validation_step(batch, _batch_idx)

    def test_epoch_end(self, results):
        """Accumulate all test batches.

        Args:
            results (dict[str, torch.Tensor]): Query IDs, resulting predictions and labels

        Returns:
            dict[str, dict[str, torch.Tensor]]: Logged metrics (MAP and MRR@k)
        """
        return self.validation_epoch_end(results)
