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
# PredictInstance = Tuple[int, str, str]
ModelInput = Any
PointwiseTrainingInput = Tuple[ModelInput, int]
PairwiseTrainingInput = Tuple[ModelInput, ModelInput]
ValidationInput = TestInput = Tuple[ModelInput, int, int]
# PredictInput = Tuple[int, ModelInput]
ModelBatch = Any
PointwiseTrainingBatch = Tuple[ModelBatch, torch.Tensor]
PairwiseTrainingBatch = Tuple[ModelBatch, ModelBatch]
ValidationBatch = TestBatch = Tuple[ModelBatch, torch.Tensor, torch.Tensor]
# PredictBatch = Tuple[torch.Tensor, ModelBatch]


class Mode(Enum):
    """Enum used to set the dataset mode."""

    POINTWISE_TRAINING = 1
    PAIRWISE_TRAINING = 2
    VALIDATION = 3
    TESTING = 4


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


class H5DataProvider(DataProvider):
    """Data provider for hdf5-based datasets (pre-processed)."""

    def __init__(self, data_dir: Path, fold_name: str) -> None:
        """Constructor.

        Args:
            data_dir (Path): Root directory of all dataset files.
            fold_name (str): Name of the fold (within data_dir) to use.
        """
        super().__init__()
        self.data_file = data_dir / "data.h5"
        self.train_file_pointwise = data_dir / fold_name / "train_pointwise.h5"
        self.train_file_pairwise = data_dir / fold_name / "train_pairwise.h5"
        self.val_file = data_dir / fold_name / "val.h5"
        self.test_file = data_dir / fold_name / "test.h5"

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

    def get_test_instance(self, index: int) -> TestInstance:
        with h5py.File(self.test_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, q_id, label

    # def get_test_instance(self, index: int) -> PredictInstance:
    #     with h5py.File(self.test_file, "r") as fp:
    #         q_id = fp["q_ids"][index]
    #         doc_id = fp["doc_ids"][index]
    #     with h5py.File(self.data_file, "r") as fp:
    #         query = fp["queries"].asstr()[q_id]
    #         doc = fp["docs"].asstr()[doc_id]
    #     return index, query, doc

    # def get_test_ids(self, index: int):
    #     with h5py.File(self.data_file, "r") as fp_data, h5py.File(
    #         self.test_file, "r"
    #     ) as fp_test:
    #         q_id = fp_test["q_ids"][index]
    #         orig_q_id = fp_data["orig_q_ids"].asstr()[q_id]
    #         doc_id = fp_test["doc_ids"][index]
    #         orig_doc_id = fp_data["orig_doc_ids"].asstr()[doc_id]
    #         return orig_q_id, orig_doc_id

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

    @property
    def num_test_instances(self) -> int:
        if not self.test_file.is_file():
            return 0
        with h5py.File(self.test_file, "r") as fp:
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

    def __getitem__(
        self, index: int
    ) -> Union[
        PointwiseTrainingInput, PairwiseTrainingInput, ValidationInput, TestInput,
    ]:
        """Return an input.

        Args:
            index (int): Item index.

        Returns:
            Union[PointwiseTrainingInput, PairwiseTrainingInput, ValidationInput, TestInput]: Input, depending on the mode.
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

    def collate_fn(
        self,
        inputs: Union[
            Iterable[PointwiseTrainingInput],
            Iterable[PairwiseTrainingInput],
            Iterable[ValidationInput],
            Iterable[TestInput],
        ],
    ) -> Union[
        PointwiseTrainingBatch, PairwiseTrainingBatch, ValidationBatch, TestBatch
    ]:
        """Collate inputs into a batch.

        Args:
            inputs (Union[Iterable[PointwiseTrainingInput], Iterable[PairwiseTrainingInput], Iterable[ValidationInput], Iterable[TestInput]]): The inputs.

        Returns:
            Union[PointwiseTrainingBatch, PairwiseTrainingBatch, ValidationBatch, TestBatch]: The resulting batch.
        """
        if self.mode == Mode.POINTWISE_TRAINING:
            model_inputs, labels = zip(*inputs)
            return self.get_model_batch(model_inputs), torch.FloatTensor(labels)

        if self.mode == Mode.PAIRWISE_TRAINING:
            pos_inputs, neg_inputs = zip(*inputs)
            return (
                self.get_model_batch(pos_inputs),
                self.get_model_batch(neg_inputs),
            )

        if self.mode in (Mode.VALIDATION, Mode.TESTING):
            model_inputs, q_ids, labels = zip(*inputs)
            return (
                self.get_model_batch(model_inputs),
                torch.LongTensor(q_ids),
                torch.LongTensor(labels),
            )


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
