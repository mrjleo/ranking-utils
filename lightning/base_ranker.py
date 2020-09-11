from collections import defaultdict
from typing import Any, Iterable, Optional, Dict, Tuple, Union

import abc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningModule

from qa_utils.lightning.sampler import DistributedQuerySampler
from qa_utils.lightning.datasets import TrainDatasetBase, ValDatasetBase
from qa_utils.lightning.metrics import average_precision, reciprocal_rank, SyncedSum


# types
# an input batch varies for each model, hence we use Any here
InputBatch = Any
ValBatch = Tuple[torch.IntTensor, InputBatch, torch.IntTensor]
LightningOutput = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


class BaseRanker(LightningModule, abc.ABC):
    def __init__(self, hparams,
                 train_ds: TrainDatasetBase, val_ds: Optional[ValDatasetBase],
                 batch_size: int, rr_k: int = 10,
                 num_workers: int = 16, uses_ddp: bool = False):
        """Abstract base class for re-rankers. Implements average precision and reciprocal rank validation.
        This class needs to be extended and (at least) the following methods must be implemented:
            * forward
            * configure_optimizers
            * training_step

        Since this class uses custom sampling in DDP mode, the `Trainer` object must be initialized using
        `replace_sampler_ddp=False` and the argument `uses_ddp=True` must be set when DDP is active.

        Args:
            hparams ([type]): All model hyperparameters
            train_ds (TrainDatasetBase): The training dataset
            val_ds (Optional[ValDatasetBase]): The validation dataset
            batch_size (int): The batch size
            rr_k (int, optional): Compute RR@K. Defaults to 10.
            num_workers (int, optional): Number of DataLoader workers. Defaults to 16.
            uses_ddp (bool, optional): Whether DDP is used. Defaults to False.
        """
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.rr_k = rr_k
        self.num_workers = num_workers
        self.uses_ddp = uses_ddp

        # used for validation in DDP mode
        # we do not normalize AP/RR as it makes no difference for validation
        self.synced_sum = SyncedSum()

    def train_dataloader(self) -> DataLoader:
        """Return a trainset DataLoader. If the trainset object has a function named `collate_fn`,
        it is used. If the model is trained in DDP mode, the standard `DistributedSampler` is used.

        Returns:
            DataLoader: The DataLoader
        """
        if self.uses_ddp:
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            shuffle = None
        else:
            sampler = None
            shuffle = True

        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, shuffle=shuffle,
                          num_workers=self.num_workers, collate_fn=getattr(self.train_ds, 'collate_fn', None))

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a validationset DataLoader if the validationset exists. If the validationset object has a function
        named `collate_fn`, it is used. If the model is validated in DDP mode, `DistributedQuerySampler` is used
        for ranking metrics to work on a query level.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no validation dataset
        """
        if self.val_ds is None:
            return None

        if self.uses_ddp:
            sampler = DistributedQuerySampler(self.val_ds)
        else:
            sampler = None

        return DataLoader(self.val_ds, batch_size=self.batch_size, sampler=sampler, shuffle=False,
                          num_workers=self.num_workers, collate_fn=getattr(self.val_ds, 'collate_fn', None))

    def validation_step(self, batch: ValBatch, batch_idx: int) -> LightningOutput:
        """Process a single validation batch.

        Args:
            batch (ValBatch): Query IDs, inputs and labels
            batch_idx (int): Batch index

        Returns:
            LightningOutput: Query IDs, resulting predictions and labels
        """
        q_ids, inputs, labels = batch
        outputs = self(*inputs)
        return {'q_ids': q_ids, 'predictions': outputs, 'labels': labels}

    def validation_epoch_end(self, results: Iterable[LightningOutput]) -> LightningOutput:
        """Accumulate all validation batches and compute AP.

        Args:
            results (Iterable[LightningOutput]): Query IDs, resulting predictions and labels

        Returns:
            LightningOutput: The sums of all APs and RRs (for each query)
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
            aps.append(average_precision(predictions, labels))
            rrs.append(reciprocal_rank(predictions, labels, self.rr_k))
        return {'log': {'val_ap_sum': self.synced_sum(torch.stack(aps)),
                        'val_rr_sum': self.synced_sum(torch.stack(rrs))}}
