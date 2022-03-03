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
ValidationInstance = TestInstance = Tuple[str, str, int, int]
PredictionInstance = Tuple[int, str, str]
Instance = Union[
    PointwiseTrainingInstance,
    PairwiseTrainingInstance,
    ValidationInstance,
    TestInstance,
    PredictionInstance,
]

ModelInput = Any
PointwiseTrainingInput = Tuple[ModelInput, int]
PairwiseTrainingInput = Tuple[ModelInput, ModelInput]
ValidationInput = TestInput = Tuple[ModelInput, int, int]
PredictionInput = Tuple[int, ModelInput]
Input = Union[
    PointwiseTrainingInput,
    PairwiseTrainingInput,
    ValidationInput,
    TestInput,
    PredictionInput,
]

ModelBatch = Any
PointwiseTrainingBatch = Tuple[ModelBatch, torch.Tensor]
PairwiseTrainingBatch = Tuple[ModelBatch, ModelBatch]
ValidationBatch = TestBatch = Tuple[ModelBatch, torch.Tensor, torch.Tensor]
PredictionBatch = Tuple[torch.Tensor, ModelBatch]
Batch = Union[
    PointwiseTrainingBatch,
    PairwiseTrainingBatch,
    ValidationBatch,
    TestBatch,
    PredictionBatch,
]


class Mode(Enum):
    """Enum used to set the dataset mode."""

    POINTWISE_TRAINING = 1
    PAIRWISE_TRAINING = 2
    VALIDATION = 100
    TESTING = 101
    PREDICTION = 102


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

    @abc.abstractmethod
    def get_test_instance(self, index: int) -> TestInstance:
        """Return a test instance.

        Args:
            index (int): The test instance index.

        Returns:
            TestInstance: Query, document, internal query ID and label.
        """
        pass

    @abc.abstractmethod
    def get_prediction_instance(self, index: int) -> PredictionInstance:
        """Return a prediction instance.

        Args:
            index (int): The prediction instance index.

        Returns:
            PredictionInstance: Index, query and document.
        """
        pass

    @abc.abstractmethod
    def get_prediction_ids(self, index: int) -> Tuple[str, str]:
        """Return the original query and document IDs for a prediction index.

        Args:
            index (int): The prediction index.

        Returns:
            Tuple[str, str]: Query ID and document ID corresponding to the index.
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

    @property
    @abc.abstractmethod
    def num_test_instances(self) -> int:
        """Return the number of test instances.

        Returns:
            int: The number of test instances.
        """
        pass

    @property
    @abc.abstractmethod
    def num_prediction_instances(self) -> int:
        """Return the number of prediction instances.

        Returns:
            int: The number of prediction instances.
        """
        pass


class H5DataProvider(DataProvider):
    """Data provider for hdf5-based datasets (pre-processed)."""

    def __init__(
        self, data_dir: Path, fold_name: str = None, predict_from: Path = None
    ) -> None:
        """Constructor.

        Args:
            data_dir (Path): Root directory of all dataset files.
            fold_name (str, optional): Name of the fold (within data_dir) to use for training. Defaults to None.
            predict_from (Path, optional): File to predict from. If None, the test set will be used. Defaults to None.
        """
        super().__init__()
        self.data_file = data_dir / "data.h5"
        if fold_name is None:
            self.train_file_pointwise = None
            self.train_file_pairwise = None
            self.val_file = None
            self.test_file = None
        else:
            self.train_file_pointwise = data_dir / fold_name / "train_pointwise.h5"
            self.train_file_pairwise = data_dir / fold_name / "train_pairwise.h5"
            self.val_file = data_dir / fold_name / "val.h5"
            self.test_file = data_dir / fold_name / "test.h5"

        if predict_from is None:
            self.pred_file = self.test_file
        else:
            self.pred_file = predict_from

    def get_pointwise_training_instance(self, index: int) -> PointwiseTrainingInstance:
        if self.train_file_pointwise is None:
            raise RuntimeError("No pointwise training instances provided")
        with h5py.File(self.train_file_pointwise, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, label

    def get_pairwise_training_instance(self, index: int) -> PairwiseTrainingInstance:
        if self.train_file_pairwise is None:
            raise RuntimeError("No pairwise training instances provided")
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
        if self.val_file is None:
            raise RuntimeError("No validation instances provided")
        with h5py.File(self.val_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, q_id, label

    def get_test_instance(self, index: int) -> TestInstance:
        if self.test_file is None:
            raise RuntimeError("No test instances provided")
        with h5py.File(self.test_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, q_id, label

    def get_prediction_instance(self, index: int) -> PredictionInstance:
        with h5py.File(self.pred_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return index, query, doc

    def get_prediction_ids(self, index: int) -> Tuple[str, str]:
        with h5py.File(self.pred_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
        with h5py.File(self.data_file, "r") as fp:
            orig_q_id = fp["orig_q_ids"].asstr()[q_id]
            orig_doc_id = fp["orig_doc_ids"].asstr()[doc_id]
            return orig_q_id, orig_doc_id

    @property
    def num_pointwise_training_instances(self) -> int:
        if self.train_file_pointwise is None or not self.train_file_pointwise.is_file():
            return 0
        with h5py.File(self.train_file_pointwise, "r") as fp:
            return len(fp["q_ids"])

    @property
    def num_pairwise_training_instances(self) -> int:
        if self.train_file_pairwise is None or not self.train_file_pairwise.is_file():
            return 0
        with h5py.File(self.train_file_pairwise, "r") as fp:
            return len(fp["q_ids"])

    @property
    def num_validation_instances(self) -> int:
        if self.val_file is None or not self.val_file.is_file():
            return 0
        with h5py.File(self.val_file, "r") as fp:
            return len(fp["q_ids"])

    @property
    def num_test_instances(self) -> int:
        if self.test_file is None or not self.test_file.is_file():
            return 0
        with h5py.File(self.test_file, "r") as fp:
            return len(fp["q_ids"])

    @property
    def num_prediction_instances(self) -> int:
        if self.pred_file is None or not self.pred_file.is_file():
            return 0
        with h5py.File(self.pred_file, "r") as fp:
            return len(fp["q_ids"])


class RankingDataset(Dataset):
    """PyTorch dataset for training, validation and testing of ranking models."""

    def __init__(
        self,
        data_provider: DataProvider,
        mode: Mode,
        get_model_input: Callable[[str, str], ModelInput],
        get_model_batch: Callable[[Iterable[ModelInput]], ModelBatch],
    ):
        """Constructor.

        Args:
            data_provider (DataProvider): A data provider.
            mode (Mode): The dataset mode, determining the instances to return.
            get_model_input (Callable[[str, str], ModelInput]): A function to create model inputs from query-document pairs.
            get_model_batch (Callable[[Iterable[ModelInput]], ModelBatch]): A function to create batches from single inputs.
        """
        self.data_provider = data_provider
        self.mode = mode
        self.get_model_input = get_model_input
        self.get_model_batch = get_model_batch

    def __getitem__(self, index: int) -> Input:
        """Return an input.

        Args:
            index (int): Item index.

        Returns:
            Input: Input, depending on the mode.
        """
        if self.mode == Mode.POINTWISE_TRAINING:
            instance = self.data_provider.get_pointwise_training_instance(index)
            query, doc, label = instance
            return self.get_model_input(query, doc), label

        if self.mode == Mode.PAIRWISE_TRAINING:
            instance = self.data_provider.get_pairwise_training_instance(index)
            query, pos_doc, neg_doc = instance
            return (
                self.get_model_input(query, pos_doc),
                self.get_model_input(query, neg_doc),
            )

        if self.mode == Mode.VALIDATION:
            instance = self.data_provider.get_validation_instance(index)
            query, doc, q_id, label = instance
            return self.get_model_input(query, doc), q_id, label

        if self.mode == Mode.TESTING:
            instance = self.data_provider.get_test_instance(index)
            query, doc, q_id, label = instance
            return self.get_model_input(query, doc), q_id, label

        if self.mode == Mode.PREDICTION:
            instance = self.data_provider.get_prediction_instance(index)
            index, query, doc = instance
            return index, self.get_model_input(query, doc)

    def __len__(self) -> int:
        """Number of instances.

        Returns:
            int: The dataset length.
        """
        if self.mode == Mode.POINTWISE_TRAINING:
            return self.data_provider.num_pointwise_training_instances

        if self.mode == Mode.PAIRWISE_TRAINING:
            return self.data_provider.num_pairwise_training_instances

        if self.mode == Mode.VALIDATION:
            return self.data_provider.num_validation_instances

        if self.mode == Mode.TESTING:
            return self.data_provider.num_test_instances

        if self.mode == Mode.PREDICTION:
            return self.data_provider.num_prediction_instances

    def collate_fn(self, inputs: Iterable[Input]) -> Batch:
        """Collate inputs into a batch.

        Args:
            inputs (Iterable[Input]): The inputs.

        Returns:
            Batch: The resulting batch.
        """
        if self.mode == Mode.POINTWISE_TRAINING:
            model_inputs, labels = zip(*inputs)
            return self.get_model_batch(model_inputs), torch.FloatTensor(labels)

        if self.mode == Mode.PAIRWISE_TRAINING:
            pos_inputs, neg_inputs = zip(*inputs)
            return self.get_model_batch(pos_inputs), self.get_model_batch(neg_inputs)

        if self.mode in (Mode.VALIDATION, Mode.TESTING):
            model_inputs, q_ids, labels = zip(*inputs)
            return (
                self.get_model_batch(model_inputs),
                torch.LongTensor(q_ids),
                torch.LongTensor(labels),
            )

        if self.mode == Mode.PREDICTION:
            indices, model_inputs = zip(*inputs)
            return torch.IntTensor(indices), self.get_model_batch(model_inputs)


class RankingDataModule(LightningDataModule, abc.ABC):
    """Data module that handles input creation, batching and data loaders.
    Must be overridden individually for each model. Methods to be implemented:
        * get_model_input
        * get_batch
    """

    def __init__(
        self,
        data_provider: DataProvider,
        training_mode: Mode,
        batch_size: int,
        num_workers: int = 16,
    ):
        """Constructor.

        Args:
            data_provider (DataProvider): A data provider.
            training_mode (Mode): The training mode to use.
            batch_size (int): The batch size to use.
            num_workers (int, optional): The number of data loader workers. Defaults to 16.
        """
        super().__init__()
        if training_mode not in (Mode.POINTWISE_TRAINING, Mode.PAIRWISE_TRAINING):
            raise ValueError(f"Invalid training mode: {training_mode}")
        self.data_provider = data_provider
        self.training_mode = training_mode
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abc.abstractmethod
    def get_model_input(self, query: str, doc: str) -> ModelInput:
        """Transform a single query-document pair into a model input.

        Args:
            query (str): The query.
            doc (str): The document.

        Returns:
            ModelInput: The model input.
        """
        pass

    @abc.abstractmethod
    def get_model_batch(self, inputs: Iterable[ModelInput]) -> ModelBatch:
        """Collate a number of model inputs into a batch.

        Args:
            inputs (Iterable[ModelInput]): The model inputs.

        Returns:
            ModelBatch: The resulting batch.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        """Return a training DataLoader.

        Returns:
            DataLoader: The DataLoader.
        """
        train_ds = RankingDataset(
            self.data_provider,
            self.training_mode,
            self.get_model_input,
            self.get_model_batch,
        )
        return DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=train_ds.collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a validation DataLoader if the validation set exists.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no validation set.
        """
        if not self.data_provider.num_validation_instances > 0:
            return None

        val_ds = RankingDataset(
            self.data_provider,
            Mode.VALIDATION,
            self.get_model_input,
            self.get_model_batch,
        )
        return DataLoader(
            val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=val_ds.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Return a test DataLoader if the test set exists.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no test dataset.
        """
        if not self.data_provider.num_test_instances > 0:
            return None

        test_ds = RankingDataset(
            self.data_provider, Mode.TESTING, self.get_model_input, self.get_model_batch
        )
        return DataLoader(
            test_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=test_ds.collate_fn,
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        """Return a prediction DataLoader if a prediction set exists.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no prediction dataset.
        """
        if not self.data_provider.num_prediction_instances > 0:
            return None

        pred_ds = RankingDataset(
            self.data_provider,
            Mode.PREDICTION,
            self.get_model_input,
            self.get_model_batch,
        )
        return DataLoader(
            pred_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=pred_ds.collate_fn,
        )
