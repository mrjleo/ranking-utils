from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
from pytorch_lightning import LightningDataModule
from ranking_utils.model import (
    PairwiseTrainingInstance,
    PointwiseTrainingInstance,
    PredictionInstance,
    TrainingMode,
    ValTestInstance,
)
from ranking_utils.model.data import (
    DataProcessor,
    PredictionDataset,
    TrainingDataset,
    ValTestDataset,
)
from torch.utils.data import DataLoader


class H5TrainingDataset(TrainingDataset):
    """Training dataset for pre-processed data (h5)."""

    def __init__(
        self,
        data_file: Path,
        train_file: Path,
        data_processor: DataProcessor,
        mode: TrainingMode,
    ) -> None:
        """Constructor.

        Args:
            data_file (Path): File that contains the corpus (.h5).
            train_file (Path): File that contains the training set (.h5).
            data_processor (DataProcessor): A model-specific data processor.
            mode (TrainingMode): The training mode, determining the inputs returned.
        """
        super().__init__(data_processor, mode)
        self.data_file = data_file
        self.train_file = train_file

    def _num_pointwise_instances(self) -> int:
        with h5py.File(self.train_file, "r") as fp:
            return len(fp["q_ids"])

    def _num_pairwise_instances(self) -> int:
        with h5py.File(self.train_file, "r") as fp:
            return len(fp["q_ids"])

    def _get_pointwise_instance(self, index: int) -> PointwiseTrainingInstance:
        with h5py.File(self.train_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, label

    def _get_pairwise_instance(self, index: int) -> PairwiseTrainingInstance:
        with h5py.File(self.train_file, "r") as fp:
            q_id = fp["q_ids"][index]
            pos_doc_id = fp["pos_doc_ids"][index]
            neg_doc_id = fp["neg_doc_ids"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            pos_doc = fp["docs"].asstr()[pos_doc_id]
            neg_doc = fp["docs"].asstr()[neg_doc_id]
        return query, pos_doc, neg_doc


class H5ValTestDataset(ValTestDataset):
    """Validation and testing dataset for pre-processed data (hdf5)."""

    def __init__(
        self, data_file: Path, val_test_file: Path, data_processor: DataProcessor
    ) -> None:
        """Constructor.

        Args:
            data_file (Path): File that contains the corpus (.h5).
            val_test_file (Path): File that contains the validation or test set (.h5).
            data_processor (DataProcessor): A model-specific data processor.
        """
        super().__init__(data_processor)
        self.data_file = data_file
        self.val_test_file = val_test_file

    def _num_instances(self) -> int:
        with h5py.File(self.val_test_file, "r") as fp:
            return len(fp["q_ids"])

    def _get_instance(self, index: int) -> ValTestInstance:
        with h5py.File(self.val_test_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
            label = fp["labels"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return query, doc, q_id, label


class H5PredictionDataset(PredictionDataset):
    """Prediction dataset for pre-processed data (hdf5)."""

    def __init__(
        self, data_file: Path, pred_file: Path, data_processor: DataProcessor
    ) -> None:
        """Constructor.

        Args:
            data_file (Path): File that contains the corpus (.h5).
            pred_file (Path): File that contains the prediction set (.h5).
            data_processor (DataProcessor): A model-specific data processor.
        """
        super().__init__(data_processor)
        self.data_file = data_file
        self.pred_file = pred_file

    def _num_instances(self) -> int:
        with h5py.File(self.pred_file, "r") as fp:
            return len(fp["q_ids"])

    def _get_instance(self, index: int) -> PredictionInstance:
        with h5py.File(self.pred_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]
        return index, query, doc

    def get_ids(self, index: int) -> Tuple[str, str]:
        with h5py.File(self.pred_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]
        with h5py.File(self.data_file, "r") as fp:
            orig_q_id = fp["orig_q_ids"].asstr()[q_id]
            orig_doc_id = fp["orig_doc_ids"].asstr()[doc_id]
            return orig_q_id, orig_doc_id


class H5DataModule(LightningDataModule):
    """Data module for H5-based datasets."""

    def __init__(
        self,
        data_dir: Union[Path, str],
        fold_name: str,
        data_processor: DataProcessor,
        batch_size: int,
        training_mode: TrainingMode = TrainingMode.POINTWISE,
        num_workers: int = 16,
    ) -> None:
        """Constructor.

        Args:
            data_dir (Union[Path, str]): Root directory of all dataset files.
            fold_name (str): Name of the fold (within `data_dir`) to use for training.
            data_processor (DataProcessor): Model-specific data processor.
            batch_size (int): The batch size to use.
            training_mode (TrainingMode, optional): The training mode to use. Defaults to TrainingMode.POINTWISE.
            num_workers (int, optional): The number of data loader workers. Defaults to 16.
        """
        super().__init__()

        if type(data_dir) != Path:
            data_dir = Path(data_dir)

        self.data_file = data_dir / "data.h5"
        self.train_file_pointwise = data_dir / fold_name / "train_pointwise.h5"
        self.train_file_pairwise = data_dir / fold_name / "train_pairwise.h5"
        self.val_file = data_dir / fold_name / "val.h5"
        self.test_file = data_dir / fold_name / "test.h5"

        self.data_processor = data_processor
        self.batch_size = batch_size
        self.training_mode = training_mode
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Return a training DataLoader.

        Returns:
            DataLoader: The DataLoader.
        """
        train_file = (
            self.train_file_pointwise
            if self.training_mode == TrainingMode.POINTWISE
            else self.train_file_pairwise
        )
        ds = H5TrainingDataset(
            self.data_file, train_file, self.data_processor, self.training_mode
        )
        return DataLoader(
            ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=ds.collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a validation DataLoader if the validation set exists.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no validation set.
        """
        if not self.val_file.is_file():
            return None

        ds = H5ValTestDataset(self.data_file, self.val_file, self.data_processor)
        return DataLoader(
            ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=ds.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Return a test DataLoader if the test set exists.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no test dataset.
        """
        if not self.test_file.is_file():
            return None

        ds = H5ValTestDataset(self.data_file, self.test_file, self.data_processor)
        return DataLoader(
            ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=ds.collate_fn,
        )
