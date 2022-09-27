import abc
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm


class TrainingSet(abc.ABC):
    """Base class for training set iterators."""

    def __init__(
        self,
        train_ids: Set[int],
        qrels: Dict[int, Dict[int, int]],
        pools: Dict[int, Set[int]],
        num_instances_per_positive: int,
        balance_queries: bool,
    ) -> None:
        """Constructor.

        Args:
            train_ids (Set[int]): Training set query IDs.
            qrels (Dict[int, Dict[int, int]]): Query IDs mapped to document IDs mapped to relevance.
            pools (Dict[int, Set[int]]): Query IDs mapped to top retrieved documents.
            num_instances_per_positive (int): Number of training instances for each relevant document.
            balance_queries (bool): Whether to balance the total number of instances for each query based on its number of positives.
        """
        self.train_ids = train_ids
        self.qrels = qrels
        self.pools = pools
        self.num_instances_per_positive = num_instances_per_positive
        self.balance_queries = balance_queries

        self.percentiles = None
        self.items = self._create_training_data()

    @abc.abstractmethod
    def _create_training_data(self) -> List[Any]:
        """Create the training data as a list containint some structure of query and document IDs.

        Returns:
            List[Any]: The training data.
        """
        pass

    @abc.abstractmethod
    def save(self, dest: Path) -> None:
        """Save the training set.

        Args:
            dest (Path): File to create.
        """
        pass

    def _get_docs_by_relevance(self, q_id: int) -> Dict[int, Set[int]]:
        """Return all documents for a query from its pool and QRels, grouped by relevance.
        Documents from the pool that have no associated relevance get a relevance of 0.

        Args:
            q_id (int): The query ID.

        Returns:
            Dict[int, Set[int]]: Relevances mapped to sets of document IDs.
        """
        result = defaultdict(set)
        for doc_id, rel in self.qrels.get(q_id, {}).items():
            result[rel].add(doc_id)
        for doc_id in self.pools.get(q_id, set()):
            rel = self.qrels[q_id].get(doc_id, 0)
            result[rel].add(doc_id)
        return result

    def _get_all_positives(self, q_id: int) -> List[int]:
        """Return all positive documents for a query from its pool and QRels.

        Args:
            q_id (int): The query ID.

        Returns:
            List[int]: A list of document IDs.
        """
        result = []
        for rel, doc_ids in self._get_docs_by_relevance(q_id).items():
            if rel > 0:
                result.extend(doc_ids)
        return result

    def _get_all_negatives(self, q_id: int) -> List[int]:
        """Return all negative documents for a query from its pool and QRels.

        Args:
            q_id (int): The query ID.

        Returns:
            List[int]: A list of document IDs.
        """
        result = []
        for rel, doc_ids in self._get_docs_by_relevance(q_id).items():
            if rel <= 0:
                result.extend(doc_ids)
        return result

    def _compute_percentiles(self) -> List[float]:
        """Compute 25%, 50% and 75% percentiles for the number of positives per query.

        Returns:
            List[float]: The percentiles.
        """
        num_positives = []
        for q_id in self.train_ids:
            num_positives.append(len(self._get_all_positives(q_id)))
        return np.percentile(num_positives, [25, 50, 75]).tolist()

    def _get_balancing_factor(self, q_id: int) -> float:
        """Return a balancing factor for a query based on its number of positives.

        Args:
            q_id (int): The query ID.

        Returns:
            float: The balancing factor.
        """
        if self.percentiles is None:
            self.percentiles = self._compute_percentiles(self)

        num_positives = len(self._get_all_positives(q_id))
        q25, q50, q75 = self.percentiles
        if num_positives < q25:
            return 5.0
        elif num_positives < q50:
            return 3.0
        elif num_positives < q75:
            return 1.5
        return 1.0

    def _get_num_instances_per_positive(self, q_id: int) -> int:
        """Return the number of instances per positive for a query. Takes balancing into account.

        Args:
            q_id (int): The query ID.

        Returns:
            int: The total number of instances for each relevant document of that query.
        """
        if self.balance_queries:
            return int(
                self.num_instances_per_positive * self._get_balancing_factor(q_id)
            )
        return self.num_instances_per_positive

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            int: The number of training instances.
        """
        return len(self.items)

    def __iter__(self) -> Iterable[Any]:
        """Yield all training examples.

        Yields:
            Tuple[Any]: The training examples.
        """
        yield from self.items


class PointwiseTrainingSet(TrainingSet):
    """A training set iterator for pointwise training instances."""

    def __init__(
        self,
        train_ids: Set[int],
        qrels: Dict[int, Dict[int, int]],
        pools: Dict[int, Set[int]],
        num_instances_per_positive: int,
        balance_queries: bool,
        balance_labels: bool,
    ) -> None:
        """Constructor.

        Args:
            train_ids (Set[int]): Training set query IDs.
            qrels (Dict[int, Dict[int, int]]): Query IDs mapped to document IDs mapped to relevance.
            pools (Dict[int, Set[int]]): Query IDs mapped to top retrieved documents.
            num_instances_per_positive (int): Number of training instances for each relevant document.
            balance_queries (bool): Whether to balance the total number of instances for each query based on its number of positives.
            balance_labels (bool): Whether to balance the total number of positives and negatives.
        """
        self.balance_labels = balance_labels
        super().__init__(
            train_ids, qrels, pools, num_instances_per_positive, balance_queries
        )

    def _create_training_data(self) -> List[Tuple[int, int, int]]:
        """Create pointwise training data as tuples of query ID, document ID, label.

        Returns:
            List[Tuple[int, int, int]]: The training data.
        """
        result = []
        # get all positives first
        for q_id in self.train_ids:
            num_instances = self._get_num_instances_per_positive(q_id)

            positives = self._get_all_positives(q_id)
            for doc_id in positives:
                if self.balance_labels:
                    result.extend([(q_id, doc_id, 1)] * num_instances)
                else:
                    result.append((q_id, doc_id, 1))

            # sample negatives
            negative_candidates = self._get_all_negatives(q_id)
            # in case there are not enough candidates
            num_neg = min(len(negative_candidates), len(positives) * num_instances)
            if num_neg > 0:
                for doc_id in random.sample(negative_candidates, num_neg):
                    result.append((q_id, doc_id, 0))
        return result

    def save(self, dest: Path) -> None:
        num_items = len(self)
        with h5py.File(dest, "w") as fp:
            ds = {
                "q_ids": fp.create_dataset("q_ids", (num_items,), dtype="int32"),
                "doc_ids": fp.create_dataset("doc_ids", (num_items,), dtype="int32"),
                "labels": fp.create_dataset("labels", (num_items,), dtype="int32"),
            }
            for i, (q_id, doc_id, label) in enumerate(
                tqdm(self, desc="Saving pointwise training set")
            ):
                ds["q_ids"][i] = q_id
                ds["doc_ids"][i] = doc_id
                ds["labels"][i] = label


class PairwiseTrainingSet(TrainingSet):
    """A training set iterator for pairwise training instances.
    Adapted from https://github.com/ucasir/NPRF/blob/6387db2fce30ee2b9f659ad1addfe949e6349f85/utils/pair_generator.py#L100.
    """

    def _create_training_data(self) -> List[Tuple[int, int, int]]:
        """Create pairwise training data as tuples of query ID, positive document ID, negative document ID.

        Returns:
            List[Tuple[int, int, int]]: The training data.
        """
        result = []
        for q_id in self.train_ids:
            # available relevances sorted in ascending order
            docs = self._get_docs_by_relevance(q_id)
            rels = sorted(docs.keys())

            num_instances = self._get_num_instances_per_positive(q_id)
            # start at 1, as the lowest relevance can not be used as positives
            for i, rel in enumerate(rels[1:], start=1):
                # take all documents with lower relevance as negative candidates
                negative_candidates = set.union(*[docs[rels[j]] for j in range(i)])
                for positive in docs[rel]:
                    sample_size = min(num_instances, len(negative_candidates))
                    negatives = random.sample(negative_candidates, sample_size)
                    result.extend(
                        zip([q_id] * sample_size, [positive] * sample_size, negatives)
                    )
        return result

    def save(self, dest: Path) -> None:
        num_items = len(self)
        with h5py.File(dest, "w") as fp:
            ds = {
                "q_ids": fp.create_dataset("q_ids", (num_items,), dtype="int32"),
                "pos_doc_ids": fp.create_dataset(
                    "pos_doc_ids", (num_items,), dtype="int32"
                ),
                "neg_doc_ids": fp.create_dataset(
                    "neg_doc_ids", (num_items,), dtype="int32"
                ),
            }
            for i, (q_id, pos_doc_id, neg_doc_id) in enumerate(
                tqdm(self, desc="Saving pairwise training set")
            ):
                ds["q_ids"][i] = q_id
                ds["pos_doc_ids"][i] = pos_doc_id
                ds["neg_doc_ids"][i] = neg_doc_id


class ContrastiveTrainingSet(TrainingSet):
    """A training set iterator for contrastive training instances."""

    def __init__(
        self,
        train_ids: Set[int],
        qrels: Dict[int, Dict[int, int]],
        pools: Dict[int, Set[int]],
        num_instances_per_positive: int,
        balance_queries: bool,
        num_negatives: int,
    ) -> None:
        """Constructor.

        Args:
            train_ids (Set[int]): Training set query IDs.
            qrels (Dict[int, Dict[int, int]]): Query IDs mapped to document IDs mapped to relevance.
            pools (Dict[int, Set[int]]): Query IDs mapped to top retrieved documents.
            num_instances_per_positive (int): Number of training instances for each relevant document.
            balance_queries (bool): Whether to balance the total number of instances for each query based on its number of positives.
            num_negatives (int): Number of negative documents to contrast each positive.
        """
        self.num_negatives = num_negatives
        super().__init__(
            train_ids, qrels, pools, num_instances_per_positive, balance_queries
        )

    def _create_training_data(self) -> List[Tuple[int, int, List[int]]]:
        """Create pairwise training data as tuples of query ID, positive document ID, negative document IDs.

        Returns:
            List[Tuple[int, int, List[int]]]: The training data.
        """
        result = []
        for q_id in self.train_ids:
            positives = self._get_all_positives(q_id)
            for positive in positives:
                negative_candidates = self._get_all_negatives(q_id)
                for _ in range(self._get_num_instances_per_positive(q_id)):
                    assert len(negative_candidates) >= self.num_negatives
                    result.append(
                        (
                            q_id,
                            positive,
                            random.sample(negative_candidates, self.num_negatives),
                        )
                    )
        return result

    def save(self, dest: Path) -> None:
        num_items = len(self)
        with h5py.File(dest, "w") as fp:
            ds = {
                "q_ids": fp.create_dataset("q_ids", (num_items,), dtype="int32"),
                "pos_doc_ids": fp.create_dataset(
                    "pos_doc_ids", (num_items,), dtype="int32"
                ),
                "neg_doc_ids": fp.create_dataset(
                    "neg_doc_ids", (num_items * self.num_negatives,), dtype="int32"
                ),
            }
            for i, (q_id, pos_doc_id, neg_doc_ids) in enumerate(
                tqdm(self, desc="Saving contrastive training set")
            ):
                ds["q_ids"][i] = q_id
                ds["pos_doc_ids"][i] = pos_doc_id
                for j, neg_doc_id in enumerate(neg_doc_ids):
                    ds["neg_doc_ids"][i * self.num_negatives + j] = neg_doc_id


class ValTestSet(object):
    """A validation/test set iterator."""

    def __init__(
        self,
        ids: Set[int],
        qrels: Dict[int, Dict[int, int]],
        pools: Dict[int, Set[int]],
    ) -> None:
        """Constructor.

        Args:
            ids (Set[int]): Validation/test set query IDs.
            qrels (Dict[int, Dict[int, int]]): Query IDs mapped to document IDs mapped to relevance.
            pools (Dict[int, Set[int]]): Query IDs mapped to top retrieved documents.
        """
        self.ids = ids
        self.qrels = qrels
        self.pools = pools

        self.items = []
        for q_id in self.ids:
            for doc_id in self.pools[q_id]:
                label = 1 if self.qrels[q_id].get(doc_id, 0) > 0 else 0
                self.items.append((q_id, doc_id, label))

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            int: Number of validation/test items.
        """
        return len(self.items)

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
        """Yield all validation/test items.

        Yields:
            Tuple[int, int, int]: Query ID, document ID, label.
        """
        yield from self.items

    def save(self, dest: Path) -> None:
        """Save the validation/test set.

        Args:
            dest (Path): File to create.
        """
        num_items = len(self)
        with h5py.File(dest, "w") as fp:
            ds = {
                "q_ids": fp.create_dataset("q_ids", (num_items,), dtype="int32"),
                "doc_ids": fp.create_dataset("doc_ids", (num_items,), dtype="int32"),
                "labels": fp.create_dataset("labels", (num_items,), dtype="int32"),
            }
            for i, (q_id, doc_id, label) in enumerate(
                tqdm(self, desc="Saving validation/test set")
            ):
                ds["q_ids"][i] = q_id
                ds["doc_ids"][i] = doc_id
                ds["labels"][i] = label


class Dataset(object):
    """Dataset class that provides iterators over training, validation and test set.

    Validation and test set contain all documents (corresponding to the query IDs in the set) from
    `pools`, which are to be re-ranked.

    Query and document IDs are converted to integers internally. Original IDs can be restored using
    `orig_q_ids` and `orig_doc_ids`.

    Query relevances may be any integer. Values greater than zero indicate positive relevance.
    """

    def __init__(
        self,
        queries: Dict[str, str],
        docs: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        pools: Dict[str, Set[str]],
    ) -> None:
        """Constructor.

        Args:
            queries (Dict[str, str]): Query IDs mapped to queries.
            docs (Dict[str, str]): Document IDs mapped to documents.
            qrels (Dict[str, Dict[str, int]]): Query IDs mapped to document IDs mapped to relevance.
            pools (Dict[str, Set[str]]): Query IDs mapped to top retrieved documents.
        """
        self.folds = []

        # assign unique integer IDs to queries and docs, but keep mappings from and to the original IDs
        self.orig_q_ids, self.int_q_ids = {}, {}
        self.queries = {}
        for i, (orig_q_id, query) in enumerate(queries.items()):
            self.orig_q_ids[i] = orig_q_id
            self.int_q_ids[orig_q_id] = i
            self.queries[i] = query

        self.orig_doc_ids, self.int_doc_ids = {}, {}
        self.docs = {}
        for i, (orig_doc_id, doc) in enumerate(docs.items()):
            self.orig_doc_ids[i] = orig_doc_id
            self.int_doc_ids[orig_doc_id] = i
            self.docs[i] = doc

        # convert qrels to integer IDs
        self.qrels = defaultdict(dict)
        for orig_q_id in qrels:
            int_q_id = self.int_q_ids[orig_q_id]
            for orig_doc_id, rel in qrels[orig_q_id].items():
                if orig_doc_id in self.int_doc_ids:
                    int_doc_id = self.int_doc_ids[orig_doc_id]
                    self.qrels[int_q_id][int_doc_id] = rel

        # convert pools to integer IDs
        self.pools = {}
        for orig_q_id, orig_doc_ids in pools.items():
            int_q_id = self.int_q_ids[orig_q_id]
            int_doc_ids = {
                self.int_doc_ids[orig_doc_id]
                for orig_doc_id in orig_doc_ids
                if orig_doc_id in self.int_doc_ids
            }
            self.pools[int_q_id] = int_doc_ids

    def add_fold(
        self, train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]
    ) -> None:
        """Add a new fold.

        Args:
            train_ids (Set[str]): Training set query IDs.
            val_ids (Set[str]): Validation set query IDs.
            test_ids (Set[str]): Test set query IDs.
        """
        # make sure no IDs are in any two sets
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

        # convert to integer IDs
        train_ids = set(map(self.int_q_ids.get, train_ids))
        val_ids = set(map(self.int_q_ids.get, val_ids))
        test_ids = set(map(self.int_q_ids.get, test_ids))

        self.folds.append((train_ids, val_ids, test_ids))

    def get_pointwise_training_set(
        self,
        fold: int,
        num_instances_per_positive: int,
        balance_queries: bool,
        balance_labels: bool,
    ) -> PointwiseTrainingSet:
        """Pointwise training set iterator for a given fold.

        Args:
            fold (int): Fold ID.
            num_instances_per_positive (int): Number of training instances for each relevant document.
            balance_queries (bool): Whether to balance the total number of instances for each query based on its number of positives.
            balance_labels (bool): Whether to balance the total number of positives and negatives.

        Returns:
            PointwiseTrainingSet: The training set.
        """
        return PointwiseTrainingSet(
            self.folds[fold][0],
            self.qrels,
            self.pools,
            num_instances_per_positive,
            balance_queries,
            balance_labels,
        )

    def get_pairwise_training_set(
        self,
        fold: int,
        num_instances_per_positive: int,
        balance_queries: bool,
    ) -> PairwiseTrainingSet:
        """Pairwise training set iterator for a given fold.

        Args:
            fold (int): Fold ID.
            num_instances_per_positive (int): Number of training instances for each relevant document.
            balance_queries (bool): Whether to balance the total number of instances for each query based on its number of positives.

        Returns:
            PairwiseTrainingSet: The training set.
        """
        return PairwiseTrainingSet(
            self.folds[fold][0],
            self.qrels,
            self.pools,
            num_instances_per_positive,
            balance_queries,
        )

    def get_contrastive_training_set(
        self,
        fold: int,
        num_instances_per_positive: int,
        balance_queries: bool,
        num_negatives: int,
    ) -> ContrastiveTrainingSet:
        """Contrastive training set iterator for a given fold.

        Args:
            fold (int): Fold ID.
            num_instances_per_positive (int): Number of training instances for each relevant document.
            balance_queries (bool): Whether to balance the total number of instances for each query based on its number of positives.
            num_negatives (int): Number of negative documents to contrast each positive.
        Returns:
            ContrastiveTrainingSet: The training set.
        """
        return ContrastiveTrainingSet(
            self.folds[fold][0],
            self.qrels,
            self.pools,
            num_instances_per_positive,
            balance_queries,
            num_negatives,
        )

    def get_val_set(self, fold: int) -> ValTestSet:
        """Validation set iterator for a given fold.

        Args:
            fold (int): Fold ID.

        Returns:
            ValTestSet: The validation set.
        """
        val_ids = self.folds[fold][1]
        return ValTestSet(val_ids, self.qrels, self.pools)

    def get_test_set(self, fold: int) -> ValTestSet:
        """Test set iterator for a given fold.

        Args:
            fold (int): Fold ID.

        Returns:
            ValTestSet: The test set.
        """
        test_ids = self.folds[fold][2]
        return ValTestSet(test_ids, self.qrels, self.pools)

    def save_collection(self, dest: Path) -> None:
        """Save the collection (queries and documents). Use the unique integer IDs for queries and documents.
        The original IDs can be recovered through a mapping that is also saved.

        Args:
            dest (Path): The file to create.
        """
        str_dt = h5py.string_dtype(encoding="utf-8")
        with h5py.File(dest, "w") as fp:
            ds = {
                "queries": fp.create_dataset(
                    "queries", (len(self.queries),), dtype=str_dt
                ),
                "orig_q_ids": fp.create_dataset(
                    "orig_q_ids", (len(self.orig_q_ids),), dtype=str_dt
                ),
                "docs": fp.create_dataset("docs", (len(self.docs),), dtype=str_dt),
                "orig_doc_ids": fp.create_dataset(
                    "orig_doc_ids", (len(self.orig_doc_ids),), dtype=str_dt
                ),
            }
            for q_id, query in tqdm(self.queries.items(), desc="Saving queries"):
                ds["queries"][q_id] = query
                ds["orig_q_ids"][q_id] = self.orig_q_ids[q_id]

            for doc_id, doc in tqdm(self.docs.items(), desc="Saving documents"):
                ds["docs"][doc_id] = doc
                ds["orig_doc_ids"][doc_id] = self.orig_doc_ids[doc_id]

    def save_qrels(self, dest: Path) -> None:
        """Save the QRels as a tab-separated file to be used with TREC-eval.
        Positive values indicate relevant documents. Zero or negative values indicate irrelevant documents.

        Args:
            dest (Path): The file to create.
        """
        with open(dest, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp, delimiter="\t")
            for q_id in self.qrels:
                for doc_id, rel in self.qrels[q_id].items():
                    orig_q_id = self.orig_q_ids[q_id]
                    orig_doc_id = self.orig_doc_ids[doc_id]
                    writer.writerow([orig_q_id, 0, orig_doc_id, rel])

    def save(
        self,
        target_dir: Path,
        num_instances_per_positive: int,
        balance_queries: bool,
        balance_labels: bool,
        num_negatives: int,
    ) -> None:
        """Save the collection, QRels and all folds of training, validation and test set.

        Args:
            target_dir (Path): Where to save the files
            num_instances_per_positive (int): Number of training instances for each relevant document.
            balance_queries (bool): Whether to balance the total number of instances for each query based on its number of positives.
            balance_labels (bool): Whether to balance the total number of positives and negatives (pointwise).
            num_negatives (int): Number of negative documents to contrast each positive (contrastive).
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        self.save_collection(target_dir / "data.h5")
        self.save_qrels(target_dir / "qrels.tsv")
        for fold in range(len(self.folds)):
            fold_dir = target_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            self.get_pointwise_training_set(
                fold, num_instances_per_positive, balance_queries, balance_labels
            ).save(fold_dir / "train_pointwise.h5")
            self.get_pairwise_training_set(
                fold, num_instances_per_positive, balance_queries
            ).save(fold_dir / "train_pairwise.h5")
            self.get_contrastive_training_set(
                fold, num_instances_per_positive, balance_queries, num_negatives
            ).save(fold_dir / "train_contrastive.h5")
            self.get_val_set(fold).save(fold_dir / "val.h5")
            self.get_test_set(fold).save(fold_dir / "test.h5")


class ParsableDataset(Dataset, abc.ABC):
    """Base class for datasets that are parsed from files."""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        """Constructor.

        Args:
            root_dir (Union[str, Path]): Directory that contains all dataset files.
        """
        self._root_dir = Path(root_dir)

        for f in self.required_files():
            if not (self.root_dir / f).is_file():
                raise RuntimeError(f"missing file: {f}")

        self.prepare_data()
        super().__init__(
            self.get_queries(), self.get_docs(), self.get_qrels(), self.get_pools()
        )
        for f in self.get_folds():
            self.add_fold(*f)

    @abc.abstractmethod
    def required_files(self) -> Iterable[Path]:
        """List all files that are required to parse this dataset. If a file is missing, an error is raised.

        Returns:
            Iterable[Path]: The requried files (relative paths).
        """
        pass

    @property
    def root_dir(self) -> Path:
        """Return the dataset directory.

        Returns:
            Path: The dataset root directory.
        """
        return self._root_dir

    def prepare_data(self) -> None:
        """This function is called in the beginning to do any preparatory work."""
        pass

    @abc.abstractmethod
    def get_queries(self) -> Dict[str, str]:
        """Return all queries.

        Returns:
            Dict[str, str]: Query IDs mapped to queries.
        """
        pass

    @abc.abstractmethod
    def get_docs(self) -> Dict[str, str]:
        """Return all documents.

        Returns:
            Dict[str, str]: Document IDs mapped to documents.
        """
        pass

    @abc.abstractmethod
    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all query relevances.

        Returns:
            Dict[str, Dict[str, int]]: Query IDs mapped to document IDs mapped to relevance.
        """
        pass

    @abc.abstractmethod
    def get_pools(self) -> Dict[str, Set[str]]:
        """Return all pools.

        Returns:
            Dict[str, Set[str]]: Query IDs mapped to top retrieved documents.
        """
        pass

    @abc.abstractmethod
    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        """Return all folds.

        Returns:
            Iterable[Tuple[Set[str], Set[str], Set[str]]]: Folds of training, validation and test query IDs.
        """
        pass
