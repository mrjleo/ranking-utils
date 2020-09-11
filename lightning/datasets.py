from typing import Any, List, Tuple

import abc
import h5py
from torch.utils.data import Dataset


# types
# an input varies for each model, hence we use Any here
SingleInput = Any
PairwiseInput = Tuple[SingleInput, SingleInput]
ValInput = Tuple[int, SingleInput, int]


class TrainDatasetBase(Dataset, abc.ABC):
    """Abstract base class for pairwise training datasets. Methods to be implemented:
        * get_single_input
        * collate_fn (optional)
    """
    @abc.abstractmethod
    def get_single_input(self, query: str, doc: str) -> SingleInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            SingleInput: The model input
        """
        pass

    def __getitem__(self, index: int) -> PairwiseInput:
        """Return a pair of positive and negative inputs for pairwise training.

        Args:
            index (int): Item index

        Returns:
            PairwiseInput: Positive and negative inputs for pairwise training
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
        with h5py.File(self.train_file, 'r') as fp:
            return len(fp['q_ids'])


class ValDatasetBase(Dataset, abc.ABC):
    """Abstract base class for pairwise training datasets. Methods to be implemented:
        * get_single_input
        * collate_fn (optional)
    """
    @abc.abstractmethod
    def get_single_input(self, query: str, doc: str) -> SingleInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            SingleInput: The model input
        """
        pass

    @property
    def offsets(self) -> List[int]:
        """Return the offsets.

        Returns:
            List[int]: The offsets
        """
        with h5py.File(self.val_file, 'r') as fp:
            return list(fp['offsets'])

    def __getitem__(self, index: int) -> ValInput:
        """Return an item.

        Args:
            index (int): Item index

        Returns:
            ValInput: Query ID, input and label
        """
        with h5py.File(self.val_file, 'r') as fp:
            q_id = fp['q_ids'][index]
            doc_id = fp['doc_ids'][index]
            label = fp['labels'][index]

        with h5py.File(self.data_file, 'r') as fp:
            query = fp['queries'][q_id]
            doc = fp['docs'][doc_id]

        # query ID does not have to be the original one here
        return q_id, self.get_single_input(query, doc), label

    def __len__(self) -> int:
        """Number of validation instances.

        Returns:
            int: The dataset length
        """
        with h5py.File(self.val_file, 'r') as fp:
            return len(fp['queries'])
