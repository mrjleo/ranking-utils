from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torchmetrics import (
    MetricCollection,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
)

PointwiseTrainingInstance = Tuple[str, str, int]
PairwiseTrainingInstance = Tuple[str, str, str]
ContrastiveTrainingInstance = Tuple[str, str, List[str]]
TrainingInstance = Union[
    PointwiseTrainingInstance, PairwiseTrainingInstance, ContrastiveTrainingInstance
]
ValTestInstance = Tuple[str, str, int, int]
PredictionInstance = Tuple[int, str, str]

ModelInput = Any
PointwiseTrainingInput = Tuple[ModelInput, int, int]
PairwiseTrainingInput = Tuple[ModelInput, ModelInput, int]
ContrastiveTrainingInput = Tuple[ModelInput, List[ModelInput], int]
TrainingInput = Union[
    PointwiseTrainingInput, PairwiseTrainingInput, ContrastiveTrainingInput
]
ValTestInput = Tuple[ModelInput, int, int]
PredictionInput = Tuple[int, ModelInput]

ModelBatch = Any
PointwiseTrainingBatch = Tuple[ModelBatch, torch.Tensor, torch.Tensor]
PairwiseTrainingBatch = Tuple[ModelBatch, ModelBatch, torch.Tensor]
ContrastiveTrainingBatch = Tuple[ModelBatch, ModelBatch, torch.Tensor]
TrainingBatch = Union[
    PointwiseTrainingBatch, PairwiseTrainingBatch, ContrastiveTrainingBatch
]
ValTestBatch = Tuple[ModelBatch, torch.Tensor, torch.Tensor]
PredictionBatch = Tuple[torch.Tensor, ModelBatch]


class TrainingMode(Enum):
    """Enum used to set the training mode."""

    POINTWISE = 0
    PAIRWISE = 1
    CONTRASTIVE = 2


class Ranker(LightningModule):
    """Base class for rankers. Implements AP, RR and nDCG for validation and testing.
    This class needs to be extended and the following methods must be implemented:
        * forward
        * configure_optimizers (alternatively, this can be implemented in the data module)
    """

    def __init__(
        self,
        training_mode: TrainingMode = TrainingMode.POINTWISE,
        pairwise_loss_margin: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            training_mode (TrainingMode, optional): How to train the model. Defaults to TrainingMode.POINTWISE.
            pairwise_loss_margin (float, optional): Margin used in pairwise loss. Defaults to 1.0.
        """
        super().__init__()
        self.training_mode = training_mode
        self.pairwise_loss_margin = pairwise_loss_margin
        self.bce = torch.nn.BCEWithLogitsLoss()

        metrics = [RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG]
        self.val_metrics = MetricCollection(
            [M(compute_on_step=False) for M in metrics],
            prefix="val_",
        )
        self.test_metrics = MetricCollection(
            [M(compute_on_step=False) for M in metrics],
            prefix="test_",
        )

    def training_step(
        self,
        batch: TrainingBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Train a single batch.

        Args:
            batch (TrainingBatch): A training batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        if self.training_mode == TrainingMode.POINTWISE:
            model_batch, labels, _ = batch
            loss = self.bce(self(model_batch).flatten(), labels.flatten())
        elif self.training_mode == TrainingMode.PAIRWISE:
            pos_model_batch, neg_model_batch, _ = batch
            pos_outputs = torch.sigmoid(self(pos_model_batch))
            neg_outputs = torch.sigmoid(self(neg_model_batch))
            loss = torch.mean(
                torch.clamp(
                    self.pairwise_loss_margin - pos_outputs + neg_outputs, min=0
                )
            )
        else:
            assert self.training_mode == TrainingMode.CONTRASTIVE
            pos_model_batch, neg_model_batch, _ = batch
            pos_outputs = torch.exp(torch.sigmoid(self(pos_model_batch)))
            neg_outputs = torch.exp(torch.sigmoid(self(neg_model_batch)))

            # split into individual negatives for each instance
            neg_outputs = neg_outputs.reshape((pos_outputs.shape[0], -1, 1)).sum(1)

            contrastive_loss = -torch.log(pos_outputs / (pos_outputs + neg_outputs))
            loss = torch.mean(contrastive_loss.flatten())

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: ValTestBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Process a validation batch. The returned query IDs are internal IDs.

        Args:
            batch (ValTestBatch): A validation batch.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, torch.Tensor]: Query IDs, scores and labels.
        """
        model_batch, q_ids, labels = batch
        return {"q_ids": q_ids, "scores": self(model_batch).flatten(), "labels": labels}

    def validation_step_end(self, step_results: Dict[str, torch.Tensor]) -> None:
        """Update the validation metrics.

        Args:
            step_results (Dict[str, torch.Tensor]): Results from a validation step.
        """
        self.val_metrics(
            step_results["scores"],
            step_results["labels"],
            indexes=step_results["q_ids"],
        )

    def validation_epoch_end(
        self, val_results: Iterable[Dict[str, torch.Tensor]]
    ) -> None:
        """Compute validation metrics.

        Args:
            val_results (Iterable[Dict[str, torch.Tensor]]): Results of the validation steps.
        """
        for metric, value in self.val_metrics.compute().items():
            self.log(metric, value, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: ValTestBatch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Process a test batch. The returned query IDs are internal IDs.

        Args:
            batch (ValTestBatch): A validation batch.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, torch.Tensor]: Query IDs, scores and labels.
        """
        model_batch, q_ids, labels = batch
        return {"q_ids": q_ids, "scores": self(model_batch).flatten(), "labels": labels}

    def test_step_end(self, step_results: Dict[str, torch.Tensor]) -> None:
        """Update the test metrics.

        Args:
            step_results (Dict[str, torch.Tensor]): Results from a test step.
        """
        self.test_metrics(
            step_results["scores"],
            step_results["labels"],
            indexes=step_results["q_ids"],
        )

    def test_epoch_end(self, test_results: Iterable[Dict[str, torch.Tensor]]) -> None:
        """Compute test metrics.

        Args:
            test_results (Iterable[Dict[str, torch.Tensor]]): Results of the test steps.
        """
        for metric, value in self.test_metrics.compute().items():
            self.log(metric, value, sync_dist=True)
        self.test_metrics.reset()

    def predict_step(
        self, batch: PredictionBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Compute scores for a prediction batch.

        Args:
            batch (PredictionBatch): Inputs.
            batch_idx (int): Batch index.
            dataloader_idx (int): DataLoader index.

        Returns:
            Dict[str, torch.Tensor]: Indices and scores.
        """
        indices, model_inputs = batch
        return {"indices": indices, "scores": self(model_inputs).flatten()}
