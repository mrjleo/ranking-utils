import abc
import itertools
from typing import Iterable, Iterator, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset

from ranking_utils.model import (
    ContrastiveTrainingInstance,
    ModelBatch,
    ModelInput,
    PairwiseTrainingInput,
    PairwiseTrainingInstance,
    PointwiseTrainingInput,
    PointwiseTrainingInstance,
    PredictionBatch,
    PredictionInput,
    PredictionInstance,
    TrainingBatch,
    TrainingInput,
    TrainingMode,
    ValTestBatch,
    ValTestInput,
    ValTestInstance,
)

TrainingInputs = Union[
    Iterable[PointwiseTrainingInput],
    Iterable[PairwiseTrainingInput],
    Iterable[PointwiseTrainingInput],
]


def subset_dataset(dataset: Dataset, size: Union[int, float, None]) -> Dataset:
    """Create a subset of a PyTorch dataset.

    The subset size may be given either as an absolute number of instances (int),
    a fraction of the data (float), or None to return the entire dataset.

    Args:
        dataset (Dataset): The dataset to create the subset of.
        size (Union[int, float, None]): Determines the subset size.

    Returns:
        Dataset: The subset.
    """
    if size is None:
        return dataset
    elif isinstance(size, int):
        indices = range(size)
    elif isinstance(size, float):
        indices = range(int(len(dataset) * size))
    return Subset(dataset, indices)


class DataProcessor(abc.ABC):
    """Abstract base class for model-specific data processors.

    Used within data modules to create model inputs and batches.
    """

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


class TrainingDataset(Dataset, abc.ABC):
    """PyTorch dataset for training of ranking models."""

    def __init__(self, data_processor: DataProcessor, mode: TrainingMode) -> None:
        """Instantiate a training dataset.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
            mode (TrainingMode): The training mode, determining the inputs returned.
        """
        self.data_processor = data_processor
        self.mode = mode

    @abc.abstractmethod
    def _num_pointwise_instances(self) -> int:
        """Return the number of pointwise training instances.

        Returns:
            int: The number of pointwise training instances.
        """
        pass

    @abc.abstractmethod
    def _num_pairwise_instances(self) -> int:
        """Return the number of pairwise training instances.

        Returns:
            int: The number of pairwise training instances.
        """
        pass

    @abc.abstractmethod
    def _num_contrastive_instances(self) -> int:
        """Return the number of contrastive training instances.

        Returns:
            int: The number of contrastive training instances.
        """
        pass

    @abc.abstractmethod
    def _get_pointwise_instance(self, index: int) -> PointwiseTrainingInstance:
        """Return the pointwise training instance corresponding to an index.

        Args:
            index (int): The index.

        Returns:
            PointwiseTrainingInstance: The corresponding training instance.
        """
        pass

    @abc.abstractmethod
    def _get_pairwise_instance(self, index: int) -> PairwiseTrainingInstance:
        """Return the pairwise training instance corresponding to an index.

        Args:
            index (int): The index.

        Returns:
            PairwiseTrainingInstance: The corresponding training instance.
        """
        pass

    @abc.abstractmethod
    def _get_contrastive_instance(self, index: int) -> ContrastiveTrainingInstance:
        """Return the contrastive training instance corresponding to an index.

        Args:
            index (int): The index.

        Returns:
            ContrastiveTrainingInstance: The corresponding training instance.
        """
        pass

    def __getitem__(self, index: int) -> TrainingInput:
        """Return a training input.

        Args:
            index (int): Item index.

        Returns:
            TrainingInput: Input, depending on the mode.
        """
        if self.mode == TrainingMode.POINTWISE:
            query, doc, label = self._get_pointwise_instance(index)
            return self.data_processor.get_model_input(query, doc), label, index

        if self.mode == TrainingMode.PAIRWISE:
            query, pos_doc, neg_doc = self._get_pairwise_instance(index)
            return (
                self.data_processor.get_model_input(query, pos_doc),
                self.data_processor.get_model_input(query, neg_doc),
                index,
            )

        if self.mode == TrainingMode.CONTRASTIVE:
            query, pos_doc, neg_docs = self._get_contrastive_instance(index)
            return (
                self.data_processor.get_model_input(query, pos_doc),
                [
                    self.data_processor.get_model_input(query, neg_doc)
                    for neg_doc in neg_docs
                ],
                index,
            )

    def __len__(self) -> int:
        """Number of training instances.

        Returns:
            int: The dataset length.
        """
        if self.mode == TrainingMode.POINTWISE:
            return self._num_pointwise_instances()

        if self.mode == TrainingMode.PAIRWISE:
            return self._num_pairwise_instances()

        if self.mode == TrainingMode.CONTRASTIVE:
            return self._num_contrastive_instances()

    def collate_fn(self, inputs: TrainingInputs) -> TrainingBatch:
        """Collate inputs into a batch.

        Args:
            inputs (TrainingInputs): The inputs.

        Returns:
            TrainingBatch: The resulting batch.
        """
        if self.mode == TrainingMode.POINTWISE:
            model_inputs, labels, indices = zip(*inputs)
            return (
                self.data_processor.get_model_batch(model_inputs),
                torch.FloatTensor(labels),
                torch.LongTensor(indices),
            )

        if self.mode == TrainingMode.PAIRWISE:
            pos_inputs, neg_inputs, indices = zip(*inputs)
            return (
                self.data_processor.get_model_batch(pos_inputs),
                self.data_processor.get_model_batch(neg_inputs),
                torch.LongTensor(indices),
            )

        if self.mode == TrainingMode.CONTRASTIVE:
            pos_inputs, neg_input_lists, indices = zip(*inputs)
            return (
                self.data_processor.get_model_batch(pos_inputs),
                self.data_processor.get_model_batch(itertools.chain(*neg_input_lists)),
                torch.LongTensor(indices),
            )


class ValTestDataset(Dataset, abc.ABC):
    """PyTorch dataset for validation and testing of ranking models."""

    def __init__(self, data_processor: DataProcessor) -> None:
        """Instantiate a validation/testing dataset.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
        """
        self.data_processor = data_processor

    @abc.abstractmethod
    def _num_instances(self) -> int:
        """Return the number of instances.

        Returns:
            int: The number of instances.
        """
        pass

    @abc.abstractmethod
    def _get_instance(self, index: int) -> ValTestInstance:
        """Return the instance corresponding to an index.

        Args:
            index (int): The index.

        Returns:
            ValTestInstance: The corresponding instance.
        """
        pass

    def __getitem__(self, index: int) -> ValTestInput:
        """Return an input.

        Args:
            index (int): Item index.

        Returns:
            ValTestInput: Input for validation or testing.
        """
        query, doc, q_id, label = self._get_instance(index)
        return self.data_processor.get_model_input(query, doc), q_id, label

    def __len__(self) -> int:
        """Number of instances.

        Returns:
            int: The dataset length.
        """
        return self._num_instances()

    def collate_fn(self, inputs: Iterable[ValTestInput]) -> ValTestBatch:
        """Collate inputs into a batch.

        Args:
            inputs (Iterable[ValTestInput]): The inputs.

        Returns:
            ValTestBatch: The resulting batch.
        """
        model_inputs, q_ids, labels = zip(*inputs)
        return (
            self.data_processor.get_model_batch(model_inputs),
            torch.LongTensor(q_ids),
            torch.LongTensor(labels),
        )


class PredictionDataset(Dataset, abc.ABC):
    """PyTorch dataset for prediction using ranking models."""

    def __init__(self, data_processor: DataProcessor) -> None:
        """Instantiate a prediction dataset.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
        """
        self.data_processor = data_processor

    @abc.abstractmethod
    def _num_instances(self) -> int:
        """Return the number of instances.

        Returns:
            int: The number of instances.
        """
        pass

    @abc.abstractmethod
    def _get_instance(self, index: int) -> PredictionInstance:
        """Return the instance corresponding to an index.

        Args:
            index (int): The index.

        Returns:
            PredictionInstance: The corresponding instance.
        """
        pass

    @abc.abstractmethod
    def ids(self) -> Iterator[Tuple[int, str, str]]:
        """Yield the original query and document IDs in the same order as the prediction instances.

        Yields:
            Tuple[int, str, str]: Index, query ID and document ID.
        """
        pass

    def __getitem__(self, index: int) -> PredictionInput:
        """Return an input.

        Args:
            index (int): Item index.

        Returns:
            PredictionInput: Input for prediction.
        """
        index, query, doc = self._get_instance(index)
        return index, self.data_processor.get_model_input(query, doc)

    def __len__(self) -> int:
        """Number of instances.

        Returns:
            int: The dataset length.
        """
        return self._num_instances()

    def collate_fn(self, inputs: Iterable[PredictionInput]) -> PredictionBatch:
        """Collate inputs into a batch.

        Args:
            inputs (Iterable[PredictionInput]): The inputs.

        Returns:
            PredictionBatch: The resulting batch.
        """
        indices, model_inputs = zip(*inputs)
        return (
            torch.IntTensor(indices),
            self.data_processor.get_model_batch(model_inputs),
        )
