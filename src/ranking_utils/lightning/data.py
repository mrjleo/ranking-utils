from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from enum import Enum

import abc
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


PointwiseTrainingInstance = Tuple[str, str, int]
PairwiseTrainingInstance = Tuple[str, str, str]
ValidationInstance = Tuple[str, str, int, int]
Input = Any
PointwiseTrainingInput = Tuple[Input, int]
PairwiseTrainingInput = Tuple[Input, Input]
ValidationInput = Tuple[Input, int, int]
Batch = Any
PointwiseTrainingBatch = Tuple[Batch, torch.Tensor]
PairwiseTrainingBatch = Tuple[Batch, Batch]
ValidationBatch = Tuple[Batch, torch.Tensor, torch.Tensor]


class TrainingMode(Enum):
    """Enum used to set the training mode."""

    POINTWISE = 1
    PAIRWISE = 2


class DataProvider(abc.ABC):
    """Abstract base class for data providers."""

    @abc.abstractmethod
    def get_pointwise_training_instance(self, index: int) -> PointwiseTrainingInstance:
        """Return a pointwise training instance.

        Args:
            index (int): The training instance index.

        Returns:
            PointwiseTrainingInstance: Query, document and label.
        """
        pass

    @abc.abstractmethod
    def get_pairwise_training_instance(self, index: int) -> PairwiseTrainingInstance:
        """Return a pairwise training instance.

        Args:
            index (int): The training instance index.

        Returns:
            PairwiseTrainingInstance: Query, positive document and negative document.
        """
        pass

    @abc.abstractmethod
    def get_validation_instance(self, index: int) -> ValidationInstance:
        """Return a validation instance.

        Args:
            index (int): The validation instance index.

        Returns:
            ValidationInstance: Query, document, internal query ID and label.
        """
        pass

    @property
    @abc.abstractmethod
    def num_pointwise_training_instances(self) -> int:
        """Return the number of pointwise training instances.

        Returns:
            int: The number of pointwise training instances.
        """
        pass

    @property
    @abc.abstractmethod
    def num_pairwise_training_instances(self) -> int:
        """Return the number of pairwise training instances.

        Returns:
            int: The number of pairwise training instances.
        """
        pass

    @property
    @abc.abstractmethod
    def num_validation_instances(self) -> int:
        """Return the number of validation instances.

        Returns:
            int: The number of validation instances.
        """
        pass


class H5DataProvider(DataProvider):
    """Data provider for hdf5-based datasets (pre-processed)."""

    def __init__(self, data_dir: Path, fold_name: str) -> None:
        """Constructor.

        Args:
            data_dir (Path): Root directory of all dataset files.
            fold_name (str): Name of the fold (within data_dir) to use.
        """
        self.data_file = data_dir / "data.h5"
        self.train_file_pointwise = data_dir / fold_name / "train_pointwise.h5"
        self.train_file_pairwise = data_dir / fold_name / "train_pairwise.h5"
        self.val_file = data_dir / fold_name / "val.h5"
        super().__init__()

    def get_pointwise_training_instance(self, index: int) -> PointwiseTrainingInstance:
        with h5py.File(self.train_file_pointwise, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, label

    def get_pairwise_training_instance(self, index: int) -> PairwiseTrainingInstance:
        with h5py.File(self.train_file_pairwise, "r") as fp:
            q_id = fp["q_ids"][index]
            pos_doc_id = fp["pos_doc_ids"][index]
            neg_doc_id = fp["neg_doc_ids"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            pos_doc = fp["docs"].asstr()[pos_doc_id]
            neg_doc = fp["docs"].asstr()[neg_doc_id]
        return query, pos_doc, neg_doc

    def get_validation_instance(self, index: int) -> ValidationInstance:
        with h5py.File(self.val_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, q_id, label

    @property
    def num_pointwise_training_instances(self) -> int:
        if not self.train_file_pointwise.is_file():
            return 0
        with h5py.File(self.train_file_pointwise, "r") as fp:
            return len(fp["q_ids"])

    @property
    def num_pairwise_training_instances(self) -> int:
        if not self.train_file_pairwise.is_file():
            return 0
        with h5py.File(self.train_file_pairwise, "r") as fp:
            return len(fp["q_ids"])

    @property
    def num_validation_instances(self) -> int:
        if not self.val_file.is_file():
            return 0
        with h5py.File(self.val_file, "r") as fp:
            return len(fp["q_ids"])


class TrainingDataset(Dataset):
    """Dataset for training."""

    def __init__(
        self,
        data_provider: DataProvider,
        training_mode: TrainingMode,
        get_input: Callable[[str, str], Input],
        get_batch: Callable[[Iterable[Input]], Batch],
    ):
        """Constructor.

        Args:
            data_provider (DataProvider): A data provider.
            training_mode (TrainingMode): The training mode, determining the training instances returned.
            get_input (Callable[[str, str], Input]): A function to create model inputs from query-document pairs.
            get_batch (Callable[[Iterable[Input]], Batch]): A function to create batches from single inputs.
        """
        self.data_provider = data_provider
        self.training_mode = training_mode
        self.get_input = get_input
        self.get_batch = get_batch

    def __getitem__(
        self, index: int
    ) -> Union[PointwiseTrainingInput, PairwiseTrainingInput]:
        """Return a training input.

        Args:
            index (int): Item index.

        Returns:
            Union[PointwiseTrainingInput, PairwiseTrainingInput]: Training input, depending on the mode.
        """
        if self.training_mode == TrainingMode.POINTWISE:
            query, doc, label = self.data_provider.get_pointwise_training_instance(
                index
            )
            return self.get_input(query, doc), label
        elif self.training_mode == TrainingMode.PAIRWISE:
            query, pos_doc, neg_doc = self.data_provider.get_pairwise_training_instance(
                index
            )
            return (
                self.get_input(query, pos_doc),
                self.get_input(query, neg_doc),
            )

    def __len__(self) -> int:
        """Number of training instances.

        Returns:
            int: The dataset length.
        """
        if self.training_mode == TrainingMode.POINTWISE:
            return self.data_provider.num_pointwise_training_instances
        elif self.training_mode == TrainingMode.PAIRWISE:
            return self.data_provider.num_pairwise_training_instances

    def collate_fn(
        self, inputs: Iterable[PointwiseTrainingInput]
    ) -> PointwiseTrainingBatch:
        """Collate inputs into a pointwise training batch.

        Args:
            inputs (Iterable[PointwiseTrainingInput]): The inputs.

        Returns:
            PointwiseTrainingBatch: A batch for pointwise training.
        """
        if self.training_mode == TrainingMode.POINTWISE:
            model_inputs, labels = zip(*inputs)
            return self.get_batch(model_inputs), torch.FloatTensor(labels)
        elif self.training_mode == TrainingMode.PAIRWISE:
            pos_inputs, neg_inputs = zip(*inputs)
            return (
                self.get_batch(pos_inputs),
                self.get_batch(neg_inputs),
            )


class ValidationDataset(Dataset):
    """Dataset for validation."""

    def __init__(
        self,
        data_provider: DataProvider,
        get_input: Callable[[str, str], Input],
        get_batch: Callable[[Iterable[Input]], Batch],
    ):
        """Constructor.

        Args:
            data_provider (DataProvider): A data provider.
            get_input (Callable[[str, str], Input]): A function to create model inputs from query-document pairs.
            get_batch (Callable[[Iterable[Input]], Batch]): A function to create batches from single inputs.
        """
        self.data_provider = data_provider
        self.get_input = get_input
        self.get_batch = get_batch

    def __getitem__(self, index: int) -> ValidationInput:
        """Return a validation input.

        Args:
            index (int): Item index.

        Returns:
            ValidationInput: Input, internal query ID and label.
        """
        query, doc, q_id, label = self.data_provider.get_validation_instance(index)
        return self.get_input(query, doc), q_id, label

    def __len__(self) -> int:
        """Number of validation instances.

        Returns:
            int: The dataset length.
        """
        return self.data_provider.num_validation_instances

    def collate_fn(self, inputs: Iterable[ValidationInput]) -> ValidationBatch:
        """Collate inputs into a pairwise training batch.

        Args:
            inputs (Iterable[ValidationInput]): The inputs.

        Returns:
            ValidationBatch: A batch for validation.
        """
        model_inputs, labels, q_ids = zip(*inputs)
        return (
            self.get_batch(model_inputs),
            torch.LongTensor(labels),
            torch.LongTensor(q_ids),
        )


class RankingDataModule(LightningDataModule, abc.ABC):
    """Data module that handles input creation, batching and data loaders.
    Must be overridden individually for each model. Methods to be implemented:
        * get_input
        * get_batch
    """

    def __init__(
        self,
        data_provider: DataProvider,
        training_mode: TrainingMode,
        batch_size: int,
        num_workers: int = 16,
    ):
        """Constructor.

        Args:
            data_provider (DataProvider): A data provider.
            training_mode (TrainingMode): The training mode to use.
            batch_size (int): The batch size to use.
            num_workers (int, optional): The number of data loader workers. Defaults to 16.
        """
        super().__init__()
        self.data_provider = data_provider
        self.training_mode = training_mode
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abc.abstractmethod
    def get_input(self, query: str, doc: str) -> Input:
        """Transform a single query-document pair into a model input.

        Args:
            query (str): The query.
            doc (str): The document.

        Returns:
            Input: The model input.
        """
        pass

    @abc.abstractmethod
    def get_batch(self, inputs: Iterable[Input]) -> Batch:
        """Collate a number of model inputs into a batch.

        Args:
            inputs (Iterable[Input]): The model inputs.

        Returns:
            Batch: The resulting batch.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        """Return a training DataLoader.

        Returns:
            DataLoader: The DataLoader.
        """
        train_ds = TrainingDataset(
            self.data_provider, self.training_mode, self.get_input, self.get_batch
        )
        return DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=getattr(train_ds, "collate_fn", None),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a validation set DataLoader if the validation set exists.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no validation set.
        """
        if self.data_provider.num_validation_instances == 0:
            return None

        val_ds = ValidationDataset(self.data_provider, self.get_input, self.get_batch)
        return DataLoader(
            val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=getattr(val_ds, "collate_fn", None),
        )

