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
    Mode,
    PointwiseTrainingBatch,
    PairwiseTrainingBatch,
    PredictionBatch,
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
        training_mode: Mode = Mode.POINTWISE_TRAINING,
        loss_margin: float = 1.0,
        hparams: Dict[str, Any] = None,
    ):
        super().__init__()
        self.training_mode = training_mode
        if hparams is not None:
            self.save_hyperparameters(hparams)

        if training_mode == Mode.POINTWISE_TRAINING:
            self.bce = torch.nn.BCEWithLogitsLoss()
        elif training_mode == Mode.PAIRWISE_TRAINING:
            self.loss_margin = loss_margin
        else:
            raise ValueError(f"Invalid training mode: {training_mode}")

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
            Sequence[str]: The metric names.
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
        if self.training_mode == Mode.POINTWISE_TRAINING:
            model_batch, labels = batch
            loss = self.bce(self(model_batch).flatten(), labels.flatten())
        elif self.training_mode == Mode.PAIRWISE_TRAINING:
            pos_model_batch, neg_model_batch = batch
            pos_outputs = torch.sigmoid(self(pos_model_batch))
            neg_outputs = torch.sigmoid(self(neg_model_batch))
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
            batch (ValidationBatch): A validation batch.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, torch.Tensor]: Query IDs, scores and labels.
        """
        model_batch, q_ids, labels = batch
        return {"q_ids": q_ids, "scores": self(model_batch).flatten(), "labels": labels}

    def validation_step_end(self, step_results: Dict[str, torch.Tensor]):
        """Update the validation metrics.

        Args:
            step_results (Dict[str, torch.Tensor]): Results from a validation step.
        """
        self.val_metrics(
            step_results["scores"],
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

    def predict_step(
        self, batch: PredictionBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Compute scores for a prediction batch.

        Args:
            batch (PredictionBatch): Inputs.
            batch_idx (int): Batch index.
            dataloader_idx (int): DataLoader index.

        Returns:
            Dict[str, torch.Tensor]: Scores and batch indices.
        """
        (model_inputs,) = batch
        return {"scores": self(model_inputs).flatten(), "batch_idx": batch_idx}
