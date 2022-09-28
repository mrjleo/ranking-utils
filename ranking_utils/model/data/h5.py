import logging
import os
import tempfile
from pathlib import Path
from re import I
from typing import Iterator, Optional, Tuple, Union

import h5py
from pytorch_lightning import LightningDataModule
from ranking_utils.model import (
    ContrastiveTrainingInstance,
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
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


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

    def _num_contrastive_instances(self) -> int:
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

    def _get_contrastive_instance(self, index: int) -> ContrastiveTrainingInstance:
        with h5py.File(self.train_file, "r") as fp:
            num_negatives = fp["neg_doc_ids"].attrs["num_negatives"]
            q_id = fp["q_ids"][index]
            pos_doc_id = fp["pos_doc_ids"][index]
            neg_doc_ids = fp["neg_doc_ids"][
                index * num_negatives : (index + 1) * num_negatives
            ]
        with h5py.File(self.data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            pos_doc = fp["docs"].asstr()[pos_doc_id]
            neg_docs = [fp["docs"].asstr()[neg_doc_id] for neg_doc_id in neg_doc_ids]
        return query, pos_doc, neg_docs


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
    """Prediction dataset for pre-processed data (H5).
    Supports test sets in H5 format and TREC runfiles in TSV format.
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        data_file: Union[Path, str],
        pred_file_h5: Union[Path, str] = None,
        pred_file_trec: Union[Path, str] = None,
    ) -> None:
        """Constructor. Exactly one prediction file must be provided.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
            data_file (Union[Path, str]): File that contains the corpus (.h5).
            pred_file_h5 (Union[Path, str], optional): File that contains the prediction set (.h5). Defaults to None.
            pred_file_trec (Union[Path, str], optional): File that contains a TREC run (.tsv). Defaults to None.
        """
        # exactly one of the files needs to be provided
        assert pred_file_h5 is not None or pred_file_trec is not None
        assert pred_file_h5 is None or pred_file_trec is None

        super().__init__(data_processor)
        self.data_file = Path(data_file)

        if pred_file_h5 is not None:
            self.pred_file = Path(pred_file_h5)
            self._temp_fd, self._temp_f = None, None

        else:
            self._temp_fd, self._temp_f = tempfile.mkstemp()
            LOGGER.info(f"storing temporary data in {self._temp_f}")

            # recover the internal integer query and doc IDs
            int_q_ids = {}
            int_doc_ids = {}
            with h5py.File(data_file, "r") as fp:
                for int_id, orig_id in enumerate(
                    tqdm(fp["orig_q_ids"].asstr(), total=len(fp["orig_q_ids"]))
                ):
                    int_q_ids[orig_id] = int_id
                for int_id, orig_id in enumerate(
                    tqdm(fp["orig_doc_ids"].asstr(), total=len(fp["orig_doc_ids"]))
                ):
                    int_doc_ids[orig_id] = int_id

            # create a test set in a temporary file
            qd_pairs = []
            with open(pred_file_trec, encoding="utf-8") as fp:
                for line in fp:
                    items = line.split()
                    qd_pairs.append((items[0], items[2]))

            with h5py.File(self._temp_f, "w") as fp:
                num_items = len(qd_pairs)
                ds = {
                    "q_ids": fp.create_dataset("q_ids", (num_items,), dtype="int32"),
                    "doc_ids": fp.create_dataset(
                        "doc_ids", (num_items,), dtype="int32"
                    ),
                    "labels": fp.create_dataset("labels", (num_items,), dtype="int32"),
                }
                for i, (q_id, doc_id) in enumerate(tqdm(qd_pairs)):
                    ds["q_ids"][i] = int_q_ids[q_id]
                    ds["doc_ids"][i] = int_doc_ids[doc_id]

            self.pred_file = self._temp_f

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

    def ids(self) -> Iterator[Tuple[int, str, str]]:
        with h5py.File(self.data_file, "r") as fp_data, h5py.File(
            self.pred_file, "r"
        ) as fp_pred:
            for i in range(len(self)):
                q_id = fp_pred["q_ids"][i]
                doc_id = fp_pred["doc_ids"][i]
                orig_q_id = fp_data["orig_q_ids"].asstr()[q_id]
                orig_doc_id = fp_data["orig_doc_ids"].asstr()[doc_id]
                yield i, orig_q_id, orig_doc_id

    def __del__(self) -> None:
        """Clean up temporary files, if any."""
        if self._temp_f is not None:
            LOGGER.info(f"removing {self._temp_f}")
            os.close(self._temp_fd)
            os.remove(self._temp_f)


class H5DataModule(LightningDataModule):
    """Data module for H5-based datasets."""

    def __init__(
        self,
        data_processor: DataProcessor,
        data_dir: Union[Path, str],
        fold_name: str,
        batch_size: int,
        training_mode: TrainingMode = TrainingMode.POINTWISE,
        num_workers: int = 16,
    ) -> None:
        """Constructor.

        Args:
            data_processor (DataProcessor): Model-specific data processor.
            data_dir (Union[Path, str]): Root directory of all dataset files.
            fold_name (str): Name of the fold (within `data_dir`) to use for training.
            batch_size (int): The batch size to use.
            training_mode (TrainingMode, optional): The training mode to use. Defaults to TrainingMode.POINTWISE.
            num_workers (int, optional): The number of data loader workers. Defaults to 16.
        """
        super().__init__()

        if type(data_dir) != Path:
            data_dir = Path(data_dir)

        self.data_file = data_dir / "data.h5"
        self.train_files = {
            TrainingMode.POINTWISE: data_dir / fold_name / "train_pointwise.h5",
            TrainingMode.PAIRWISE: data_dir / fold_name / "train_pairwise.h5",
            TrainingMode.CONTRASTIVE: data_dir / fold_name / "train_contrastive.h5",
        }
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
        ds = H5TrainingDataset(
            self.data_file,
            self.train_files[self.training_mode],
            self.data_processor,
            self.training_mode,
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
