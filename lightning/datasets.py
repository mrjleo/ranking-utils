from pathlib import Path
from typing import Any, Tuple

import abc
import h5py
from torch.utils.data import Dataset


# inputs vary for each model, hence we use Any here
Input = Any
TrainingInput = Tuple[Input, Input]
ValTestInput = Tuple[int, int, Input, int]


class TrainDatasetBase(Dataset, abc.ABC):
    """Abstract base class for pairwise training datasets. Methods to be implemented:
        * get_single_input
        * collate_fn (optional)

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
    """
    def __init__(self, data_file: Path, train_file: Path):
        self.data_file = data_file
        self.train_file = train_file

        with h5py.File(train_file, 'r') as fp:
            self.length = len(fp['q_ids'])

    @abc.abstractmethod
    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        pass

    def __getitem__(self, index: int) -> TrainingInput:
        """Return a pair of positive and negative inputs for pairwise training.

        Args:
            index (int): Item index

        Returns:
            TrainingInput: Positive and negative inputs for pairwise training
        """
        with h5py.File(self.train_file, 'r') as fp:
            q_id = fp['q_ids'][index]
            pos_doc_id = fp['pos_doc_ids'][index]
            neg_doc_id = fp['neg_doc_ids'][index]

        with h5py.File(self.data_file, 'r') as fp:
            query = fp['queries'][q_id]
            pos_doc = fp['docs'][pos_doc_id]
            neg_doc = fp['docs'][neg_doc_id]

        return self.get_single_input(query, pos_doc), self.get_single_input(query, neg_doc)

    def __len__(self) -> int:
        """Number of training instances.

        Returns:
            int: The dataset length
        """
        return self.length


class ValTestDatasetBase(Dataset, abc.ABC):
    """Abstract base class for validation/testing datasets. Methods to be implemented:
        * get_single_input
        * collate_fn (optional)

    The datasets yields internal integer query IDs that can be held by tensors. The original IDs can be recovered using `orig_q_ids`.

    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validation-/testset file
    """
    def __init__(self, data_file: Path, val_test_file: Path):
        self.data_file = data_file
        self.val_test_file = val_test_file

        with h5py.File(data_file, 'r') as fp:
            self.orig_q_ids = list(fp['orig_q_ids'])
            self.orig_doc_ids = list(fp['orig_doc_ids'])

        with h5py.File(val_test_file, 'r') as fp:
            self.offsets = list(fp['offsets'])
            self.length = len(fp['queries'])

    @abc.abstractmethod
    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        pass

    def __getitem__(self, index: int) -> ValTestInput:
        """Return an item.

        Args:
            index (int): Item index

        Returns:
            ValTestInput: Query ID, input and label
        """
        with h5py.File(self.val_test_file, 'r') as fp:
            q_id = fp['q_ids'][index]
            doc_id = fp['doc_ids'][index]
            label = fp['labels'][index]

        with h5py.File(self.data_file, 'r') as fp:
            query = fp['queries'][q_id]
            doc = fp['docs'][doc_id]

        # return the internal query and document IDs here
        return q_id, doc_id, self.get_single_input(query, doc), label

    def __len__(self) -> int:
        """Number of validation/testing instances.

        Returns:
            int: The dataset length
        """
        return self.length
