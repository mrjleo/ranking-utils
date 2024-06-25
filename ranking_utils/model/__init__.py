from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG

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
        margin: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            training_mode (TrainingMode, optional): How to train the model. Defaults to TrainingMode.POINTWISE.
            margin (float, optional): Margin used in pairwise loss. Defaults to 1.0.
        """
        super().__init__()
        self.training_mode = training_mode
        self.margin = margin
        self.bce = torch.nn.BCEWithLogitsLoss()

        metrics = [RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG]
        self.val_metrics = MetricCollection(
            [M() for M in metrics],
            prefix="val",
        )
        self.test_metrics = MetricCollection(
            [M() for M in metrics],
            prefix="test",
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
                torch.clamp(self.margin - pos_outputs + neg_outputs, min=0)
            )
        else:
            assert self.training_mode == TrainingMode.CONTRASTIVE
            pos_model_batch, neg_model_batch, _ = batch
            pos_outputs = torch.exp(self(pos_model_batch)).squeeze(-1)
            neg_outputs = torch.exp(self(neg_model_batch))

            # split into individual negatives for each instance
            # divide each positive score by itself and the corresponding negatives
            neg_outputs_split = neg_outputs.reshape((pos_outputs.shape[0], -1))
            contrastive_loss = -torch.log(
                pos_outputs / (pos_outputs + neg_outputs_split.sum(1))
            )
            loss = torch.mean(contrastive_loss.flatten())

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: ValTestBatch, batch_idx: int) -> None:
        """Process a validation batch and update metrics.

        Args:
            batch (ValTestBatch): A validation batch.
            batch_idx (int): Batch index.
        """
        model_batch, q_ids, labels = batch
        self.val_metrics(
            self(model_batch).flatten(),
            labels,
            indexes=q_ids,
        )

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics."""
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: ValTestBatch, batch_idx: int) -> None:
        """Process a test batch and update metrics.

        Args:
            batch (ValTestBatch): A validation batch.
            batch_idx (int): Batch index.
        """
        model_batch, q_ids, labels = batch
        self.test_metrics(
            self(model_batch).flatten(),
            labels,
            indexes=q_ids,
        )

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics."""
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
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
