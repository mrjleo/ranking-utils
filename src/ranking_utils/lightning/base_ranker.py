import abc
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG

from ranking_utils.lightning.datasets import PointwiseTrainDatasetBase, PairwiseTrainDatasetBase, ValTestDatasetBase


# input batches vary for each model, hence we use Any here
InputBatch = Any
PointwiseTrainBatch = Tuple[InputBatch, torch.FloatTensor]
PairwiseTrainBatch = Tuple[InputBatch, InputBatch]
ValTestBatch = Tuple[torch.LongTensor, torch.LongTensor, InputBatch, torch.LongTensor]


class BaseRanker(LightningModule, abc.ABC):
    """Abstract base class for re-rankers. Implements average precision and reciprocal rank validation.
    This class needs to be extended and (at least) the following methods must be implemented:
        * forward
        * configure_optimizers

    Args:
        hparams (Dict[str, Any]): All model hyperparameters
        train_ds (Union[PointwiseTrainDatasetBase, PairwiseTrainDatasetBase]): The training dataset
        val_ds (Optional[ValTestDatasetBase]): The validation dataset
        test_ds (Optional[ValTestDatasetBase]): The testing dataset
        loss_margin (float, optional): Margin used in pairwise loss
        batch_size (int): The batch size
        num_workers (int, optional): Number of DataLoader workers. Defaults to 16.
    """
    def __init__(self, hparams: Dict[str, Any],
                 train_ds: Union[PointwiseTrainDatasetBase, PairwiseTrainDatasetBase],
                 val_ds: Optional[ValTestDatasetBase], test_ds: Optional[ValTestDatasetBase],
                 loss_margin: Optional[float],
                 batch_size: int,
                 num_workers: int = 16):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.loss_margin = loss_margin
        self.batch_size = batch_size
        self.num_workers = num_workers
        if issubclass(train_ds.__class__, PointwiseTrainDatasetBase):
            self.training_mode = 'pointwise'
            self.bce = torch.nn.BCEWithLogitsLoss()
        elif issubclass(train_ds.__class__, PairwiseTrainDatasetBase):
            self.training_mode = 'pairwise'
        else:
            self.training_mode = None

        self.val_metrics = MetricCollection([
            RetrievalMAP(compute_on_step=False),
            RetrievalMRR(compute_on_step=False),
            RetrievalNormalizedDCG(compute_on_step=False)
        ], prefix='val_')

    @property
    def val_metric_names(self) -> Sequence[str]:
        """Return all validation metrics that are computed after each epoch.

        Returns:
            Sequence[str]: The metric names
        """
        return self.val_metrics.keys()

    def train_dataloader(self) -> DataLoader:
        """Return a trainset DataLoader. If the trainset object has a function named `collate_fn`,
        it is used.

        Returns:
            DataLoader: The DataLoader
        """
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=getattr(self.train_ds, 'collate_fn', None))

    def training_step(self, batch: Union[PointwiseTrainBatch, PairwiseTrainBatch], batch_idx: int) -> torch.Tensor:
        """Train a single batch.

        Args:
            batch (Union[PointwiseTrainBatch, PairwiseTrainBatch]): A training batch, depending on the mode
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Training loss
        """
        if self.training_mode == 'pointwise':
            inputs, labels = batch
            loss = self.bce(self(inputs).flatten(), labels.flatten())
        elif self.training_mode == 'pairwise':
            pos_inputs, neg_inputs = batch
            pos_outputs = torch.sigmoid(self(pos_inputs))
            neg_outputs = torch.sigmoid(self(neg_inputs))
            loss = torch.mean(torch.clamp(self.loss_margin - pos_outputs + neg_outputs, min=0))
        else:
            raise RuntimeError('Unsupported training dataset (should subclass PointwiseTrainDatasetBase or PairwiseTrainDatasetBase)')
        self.log('train_loss', loss)
        return loss

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a validationset DataLoader if the validationset exists. If the validationset object has a function
        named `collate_fn`, it is used.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no validation dataset
        """
        if self.val_ds is None:
            return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=getattr(self.val_ds, 'collate_fn', None))

    def validation_step(self, batch: ValTestBatch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Process a single validation batch. The returned query IDs are internal integer IDs.

        Args:
            batch (ValTestBatch): Query IDs, document IDs, inputs and labels
            batch_idx (int): Batch index

        Returns:
            Dict[str, torch.Tensor]: Query IDs, predictions and labels
        """
        q_ids, _, inputs, labels = batch
        return {'q_ids': q_ids, 'predictions': self(inputs).flatten(), 'labels': labels}

    def validation_step_end(self, step_results: Dict[str, torch.Tensor]):
        """Update the validation metrics.

        Args:
            step_results (Dict[str, torch.Tensor]): Results from a single validation step
        """
        self.val_metrics(step_results['predictions'], step_results['labels'], indexes=step_results['q_ids'])

    def validation_epoch_end(self, val_results: Iterable[Dict[str, torch.Tensor]]):
        """Compute validation metrics. The results may be approximate.

        Args:
            val_results (Iterable[Dict[str, torch.Tensor]]): Results of validation steps
        """
        for metric, value in self.val_metrics.compute().items():
            self.log(metric, value, sync_dist=True)
        self.val_metrics.reset()

    def predict_dataloader(self) -> Optional[DataLoader]:
        """Return a testset DataLoader if the testset exists. If the testset object has a function
        named `collate_fn`, it is used.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no testing dataset
        """
        if self.test_ds is None:
            return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=getattr(self.test_ds, 'collate_fn', None))

    def predict_step(self, batch: ValTestBatch, batch_idx: int, dataloader_idx: int) -> Dict[str, torch.Tensor]:
        """Predict a single batch. The returned query and document IDs are internal integer IDs.

        Args:
            batch (ValTestBatch): Query IDs, document IDs, inputs and labels
            batch_idx (int): Batch index
            dataloader_idx (int): DataLoader index

        Returns:
            Dict[str, torch.Tensor]: Query IDs, document IDs and predictions
        """
        q_ids, doc_ids, inputs, _ = batch
        return {
            'q_ids': q_ids,
            'doc_ids': doc_ids,
            'predictions': self(inputs).flatten()
        }