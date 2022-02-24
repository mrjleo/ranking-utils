import abc
from typing import Any, Dict, Iterable, Sequence, Union

import torch
from pytorch_lightning import LightningModule
from torchmetrics import (
    MetricCollection,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
)

from ranking_utils.lightning.data import (
    TrainingMode,
    PointwiseTrainingBatch,
    PairwiseTrainingBatch,
    ValidationBatch,
)


class Ranker(LightningModule, abc.ABC):
    """Base class for rankers. Implements AP, RR and nDCG validation.
    This class needs to be extended and the following methods must be implemented:
        * forward
        * configure_optimizers (alternatively, this can be implemented in the data module)
    """

    def __init__(
        self,
        training_mode: TrainingMode = TrainingMode.POINTWISE,
        loss_margin: float = 1.0,
        hparams: Dict[str, Any] = None,
    ):
        super().__init__()
        self.training_mode = training_mode
        if hparams is not None:
            self.save_hyperparameters(hparams)

        if training_mode == TrainingMode.POINTWISE:
            self.bce = torch.nn.BCEWithLogitsLoss()
        elif training_mode == TrainingMode.PAIRWISE:
            self.loss_margin = loss_margin
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        self.val_metrics = MetricCollection(
            [
                RetrievalMAP(compute_on_step=False),
                RetrievalMRR(compute_on_step=False),
                RetrievalNormalizedDCG(compute_on_step=False),
            ],
            prefix="val_",
        )

    @property
    def val_metric_names(self) -> Sequence[str]:
        """Return all validation metrics that are computed after each epoch.

        Returns:
            Sequence[str]: The metric names
        """
        return self.val_metrics.keys()

    def training_step(
        self,
        batch: Union[PointwiseTrainingBatch, PairwiseTrainingBatch],
        batch_idx: int,
    ) -> torch.Tensor:
        """Train a single batch.

        Args:
            batch (Union[PointwiseTrainingBatch, PairwiseTrainingBatch]): A training batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        if self.training_mode == TrainingMode.POINTWISE:
            inputs, labels = batch
            loss = self.bce(self(inputs).flatten(), labels.flatten())
        elif self.training_mode == TrainingMode.PAIRWISE:
            pos_inputs, neg_inputs = batch
            pos_outputs = torch.sigmoid(self(pos_inputs))
            neg_outputs = torch.sigmoid(self(neg_inputs))
            loss = torch.mean(
                torch.clamp(self.loss_margin - pos_outputs + neg_outputs, min=0)
            )

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: ValidationBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Process a validation batch. The returned query IDs are internal IDs.

        Args:
            batch (ValidationBatch): Inputs, internal query IDs and labels.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, torch.Tensor]: Query IDs, predictions and labels.
        """
        inputs, q_ids, labels = batch
        return {"q_ids": q_ids, "predictions": self(inputs).flatten(), "labels": labels}

    def validation_step_end(self, step_results: Dict[str, torch.Tensor]):
        """Update the validation metrics.

        Args:
            step_results (Dict[str, torch.Tensor]): Results from a single validation step.
        """
        self.val_metrics(
            step_results["predictions"],
            step_results["labels"],
            indexes=step_results["q_ids"],
        )

    def validation_epoch_end(self, val_results: Iterable[Dict[str, torch.Tensor]]):
        """Compute validation metrics. The results may be approximate.

        Args:
            val_results (Iterable[Dict[str, torch.Tensor]]): Results of the validation steps.
        """
        for metric, value in self.val_metrics.compute().items():
            self.log(metric, value, sync_dist=True)
        self.val_metrics.reset()
