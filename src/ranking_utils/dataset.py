import abc
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple, List, Iterable

import h5py
import numpy as np
from tqdm import tqdm


class PointwiseTrainingset(object):
    """A trainingset iterator for pointwise training.
    Negatives are sampled randomly from the corresponding query pools.

    Args:
        train_ids (Set[int]): Trainset query IDs
        qrels (Dict[int, Dict[int, int]]): Query IDs mapped to document IDs mapped to relevance
        pools (Dict[int, Set[int]]): Query IDs mapped to top retrieved documents
        num_negatives (int): Number of negatives per positive
    """
    def __init__(self, train_ids: Set[int], qrels: Dict[int, Dict[int, int]], pools: Dict[int, Set[int]],
                 num_negatives: int):
        self.train_ids = train_ids
        self.qrels = qrels
        self.pools = pools
        self.num_negatives = num_negatives
        self.trainset = self._create_trainset()

    def _create_trainset(self) -> List[Tuple[int, int, int]]:
        """Create the trainingset as tuples of query ID, document ID, label.

        Returns:
            List[Tuple[int, int, int]]: The trainingset
        """
        result = []
        # get all positives first
        positives = defaultdict(list)
        for q_id in self.train_ids:
            for doc_id, rel in self.qrels[q_id].items():
                if rel > 0:
                    positives[q_id].append(doc_id)
                    result.extend([(q_id, doc_id, 1)] * self.num_negatives)

        # sample negatives
        for q_id in self.train_ids:
            # all documents from the pool with no positive relevance
            candidates = [doc_id for doc_id in self.pools.get(q_id, []) if self.qrels[q_id].get(doc_id, 0) <= 0]
            # in case there are not enough candidates
            num_neg = min(len(candidates), len(positives[q_id]) * self.num_negatives)
            if num_neg > 0:
                for doc_id in random.sample(candidates, num_neg):
                    result.append((q_id, doc_id, 0))

        for q_id, _, _ in result:
            assert q_id in self.train_ids

        return result

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            int: The number of training instances
        """
        return len(self.trainset)

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
        """Yield all training examples.

        Yields:
            Tuple[int, int, int]: Query ID, document ID, label
        """
        yield from self.trainset

    def save(self, dest: Path):
        """Save the pointwise trainingset.

        Args:
            dest (Path): File to create
        """
        num_items = len(self)
        with h5py.File(dest, 'w') as fp:
            ds = {
                'q_ids': fp.create_dataset('q_ids', (num_items,), dtype='int32'),
                'doc_ids': fp.create_dataset('doc_ids', (num_items,), dtype='int32'),
                'labels': fp.create_dataset('labels', (num_items,), dtype='int32')
            }
            for i, (q_id, doc_id, label) in enumerate(tqdm(self, desc='Saving pointwise trainset')):
                ds['q_ids'][i] = q_id
                ds['doc_ids'][i] = doc_id
                ds['labels'][i] = label


class PairwiseTrainingset(object):
    """A trainingset iterator for pairwise training.
    The number of examples per query is balanced based on its number of positives.

    Args:
        train_ids (Set[int]): Trainset query IDs
        qrels (Dict[int, Dict[int, int]]): Query IDs mapped to document IDs mapped to relevance
        pools (Dict[int, Set[int]]): Query IDs mapped to top retrieved documents
        num_negatives (int): Number of negatives per positive
        query_limit (int): Maximum number of training examples per query
    """
    def __init__(self, train_ids: Set[int], qrels: Dict[int, Dict[int, int]], pools: Dict[int, Set[int]],
                 num_negatives: int, query_limit: int):
        self.train_ids = train_ids
        self.qrels = qrels
        self.pools = pools
        self.num_negatives = num_negatives
        self.query_limit = query_limit

        self.percentiles = self._compute_percentiles()
        self.trainset = self._create_trainset()

    def _get_docs_by_relevance(self, q_id: int) -> Dict[int, Set[int]]:
        """Return all documents for a query from its pool and qrels, grouped by relevance.
        Documents from the pool that have no associated relevance get a relevance of 0.

        Args:
            q_id (int): The query ID

        Returns:
            Dict[int, Set[int]]: Relevances mapped to sets of document IDs
        """
        result = defaultdict(set)
        for doc_id, rel in self.qrels.get(q_id, {}).items():
            result[rel].add(doc_id)
        for doc_id in self.pools.get(q_id, set()):
            rel = self.qrels[q_id].get(doc_id, 0)
            result[rel].add(doc_id)
        return result

    def _get_all_positives(self, q_id: int) -> List[int]:
        """Return all positive documents for a query.

        Args:
            q_id (int): The query ID

        Returns:
            List[int]: A list of document IDs
        """
        result = []
        for rel, doc_ids in self._get_docs_by_relevance(q_id).items():
            if rel > 0:
                result.extend(doc_ids)
        return result

    def _compute_percentiles(self) -> List[float]:
        """Compute 25%, 50% and 75% percentiles for the number of positives per query.

        Returns:
            List[float]: The percentiles
        """
        num_positives = []
        for q_id in self.train_ids:
            num_positives.append(len(self._get_all_positives(q_id)))
        return np.percentile(num_positives, [25, 50, 75]).tolist()

    def _get_balancing_factor(self, q_id: int) -> float:
        """Return a balancing factor for a query based on its number of positives.

        Args:
            q_id (int): The query ID

        Returns:
            float: The balancing factor
        """
        num_positives = len(self._get_all_positives(q_id))
        q25, q50, q75 = self.percentiles
        if num_positives < q25:
            return 5.0
        elif num_positives < q50:
            return 3.0
        elif num_positives < q75:
            return 1.5
        return 1.0

    def _get_triples(self, q_id: int) -> List[Tuple[int, int, int]]:
        """Return all training triples for a query as tuples of query ID, positive document ID, negative document ID.

        Args:
            q_id (int): The query ID

        Returns:
            List[Tuple[int, int, int]]: A list of training triples
        """
        docs = self._get_docs_by_relevance(q_id)
        result = []

        # balance the number of pairs for this query, based on the total number of positives
        factor = self._get_balancing_factor(q_id)
        num_negatives = int(self.num_negatives * factor)
        query_limit = int(self.query_limit * factor)

        # available relevances sorted in ascending order
        rels = sorted(docs.keys())

        # start at 1, as the lowest relevance can not be used as positives
        for i, rel in enumerate(rels[1:], start=1):
            # take all documents with lower relevance as negative candidates
            negative_candidates = set.union(*[docs[rels[j]] for j in range(i)])
            for positive in docs[rel]:
                sample_size = min(num_negatives, len(negative_candidates))
                negatives = random.sample(negative_candidates, sample_size)
                result.extend(zip([q_id] * sample_size, [positive] * sample_size, negatives))

        for q_id, _, _ in result:
            assert q_id in self.train_ids

        if len(result) > query_limit:
            return random.sample(result, query_limit)
        return result

    def _create_trainset(self) -> List[Tuple[int, int, int]]:
        """Create the trainingset as tuples of query ID, positive document ID, negative document ID.

        Returns:
            List[Tuple[int, int, int]]: The trainingset
        """
        result = []
        for q_id in self.train_ids:
            result.extend(self._get_triples(q_id))
        return result

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            int: The number of training pairs
        """
        return len(self.trainset)

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
        """Yield all training examples.

        Yields:
            Tuple[int, int, int]: Query ID, positive document ID, negative document ID
        """
        yield from self.trainset

    def save(self, dest: Path):
        """Save the pairwise trainingset.

        Args:
            dest (Path): File to create
        """
        num_items = len(self)
        with h5py.File(dest, 'w') as fp:
            ds = {
                'q_ids': fp.create_dataset('q_ids', (num_items,), dtype='int32'),
                'pos_doc_ids': fp.create_dataset('pos_doc_ids', (num_items,), dtype='int32'),
                'neg_doc_ids': fp.create_dataset('neg_doc_ids', (num_items,), dtype='int32')
            }
            for i, (q_id, pos_doc_id, neg_doc_id) in enumerate(tqdm(self, desc='Saving pairwise trainset')):
                ds['q_ids'][i] = q_id
                ds['pos_doc_ids'][i] = pos_doc_id
                ds['neg_doc_ids'][i] = neg_doc_id


class Testset(object):
    """A testset iterator.

    Args:
        test_ids (Set[int]): Testset query IDs
        qrels (Dict[int, Dict[int, int]]): Query IDs mapped to document IDs mapped to relevance
        pools (Dict[int, Set[int]]): Query IDs mapped to top retrieved documents

    Yields:
        Tuple[int, int, int]: Query ID, document ID, label
    """
    def __init__(self, test_ids: Set[int], qrels: Dict[int, Dict[int, int]], pools: Dict[int, Set[int]]):
        self.test_ids = test_ids
        self.qrels = qrels
        self.pools = pools
        self.testset = self._create_testset()

    def _create_testset(self) -> List[Tuple[int, int, int]]:
        """Create a set of documents for the query IDs. For each query, the set contains its pool.

        Returns:
            List[Tuple[int, int, int]]: Tuples containing query ID, document ID and label
        """
        result = []
        for q_id in self.test_ids:
            for doc_id in self.pools[q_id]:
                    label = 1 if self.qrels[q_id].get(doc_id, 0) > 0 else 0
                    result.append((q_id, doc_id, label))

        for q_id, _, _ in result:
            assert q_id in self.test_ids

        return result

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            int: Number of test items
        """
        return len(self.testset)

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
        """Yield all test items.

        Yields:
            Tuple[int, int, int]: Query ID, document ID, label
        """
        yield from self.testset

    def save(self, dest: Path):
        """Save the testset.

        Args:
            dest (Path): File to create
        """
        num_items = len(self)
        with h5py.File(dest, 'w') as fp:
            ds = {
                'q_ids': fp.create_dataset('q_ids', (num_items,), dtype='int32'),
                'doc_ids': fp.create_dataset('doc_ids', (num_items,), dtype='int32'),
                'labels': fp.create_dataset('labels', (num_items,), dtype='int32'),
                'offsets': fp.create_dataset('offsets', (len(self.test_ids),), dtype='int32')
            }
            last_q_id = None
            i_offsets = 0
            for i, (q_id, doc_id, label) in enumerate(tqdm(self, desc='Saving testset')):
                ds['q_ids'][i] = q_id
                ds['doc_ids'][i] = doc_id
                ds['labels'][i] = label

                # offsets are only used for validation in DDP mode
                if q_id != last_q_id:
                    ds['offsets'][i_offsets] = i
                    last_q_id = q_id
                    i_offsets += 1
            assert i_offsets == len(self.test_ids)


class Dataset(object):
    """Dataset class that provides iterators over train-, val- and testset.

    Validation- and testset contain all documents (corresponding to the query IDs in the set) from
    `pools`, which are to be re-ranked.

    Query and document IDs are converted to integers internally. Original IDs can be restored using
    `orig_q_ids` and `orig_doc_ids`.

    Query relevances may be any integer. Values greater than zero indicate positive relevance.

    Args:
        queries (Dict[str, str]): Query IDs mapped to queries
        docs (Dict[str, str]): Document IDs mapped to documents
        qrels (Dict[str, Dict[str, int]]): Query IDs mapped to document IDs mapped to relevance
        pools (Dict[str, Set[str]]): Query IDs mapped to top retrieved documents
    """
    def __init__(self,
                 queries: Dict[str, str], docs: Dict[str, str],
                 qrels: Dict[str, List[Tuple[str, int]]],
                 pools: Dict[str, Set[str]]):
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
            int_doc_ids = {self.int_doc_ids[orig_doc_id] for orig_doc_id in orig_doc_ids if orig_doc_id in self.int_doc_ids}
            self.pools[int_q_id] = int_doc_ids

    def add_fold(self, train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]):
        """Add a new fold.

        Args:
            train_ids (Set[str]): Trainset query IDs
            val_ids (Set[str]): Validationset query IDs
            test_ids (Set[str]): Testset query IDs
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

    def get_pointwise_trainingset(self, fold: int, num_negatives: int) -> PointwiseTrainingset:
        """Pointwise trainingset iterator for a given fold.

        Args:
            fold (int): Fold ID
            num_negatives (int): Number of negatives per positive

        Returns:
            PointwiseTrainingset: The trainingset
        """
        train_ids = self.folds[fold][0]
        return PointwiseTrainingset(train_ids, self.qrels, self.pools, num_negatives)

    def get_pairwise_trainingset(self, fold: int, num_negatives: int, query_limit: int) -> PairwiseTrainingset:
        """Pairwise trainingset iterator for a given fold.

        Args:
            fold (int): Fold ID
            num_negatives (int): Number of negatives per positive
            query_limit (int): Maximum number of training examples per query

        Returns:
            PairwiseTrainingset: The trainingset
        """
        train_ids = self.folds[fold][0]
        return PairwiseTrainingset(train_ids, self.qrels, self.pools, num_negatives, query_limit)

    def get_valset(self, fold: int) -> Testset:
        """Validationset iterator for a given fold.

        Args:
            fold (int): Fold ID

        Returns:
            Testset: The validationset
        """
        val_ids = self.folds[fold][1]
        return Testset(val_ids, self.qrels, self.pools)

    def get_testset(self, fold: int) -> Testset:
        """Testset iterator for a given fold.

        Args:
            fold (int): Fold ID

        Returns:
            Testset: The testset
        """
        test_ids = self.folds[fold][2]
        return Testset(test_ids, self.qrels, self.pools)

    def save_collection(self, dest: Path):
        """Save the collection (queries and documents). Use the unique integer IDs for queries and documents.
        The original IDs can be recovered through a mapping that is also saved.

        Args:
            dest (Path): The file to create
        """
        str_dt = h5py.string_dtype(encoding='utf-8')
        with h5py.File(dest, 'w') as fp:
            ds = {
                'queries': fp.create_dataset('queries', (len(self.queries),), dtype=str_dt),
                'orig_q_ids': fp.create_dataset('orig_q_ids', (len(self.orig_q_ids),), dtype=str_dt),
                'docs': fp.create_dataset('docs', (len(self.docs),), dtype=str_dt),
                'orig_doc_ids': fp.create_dataset('orig_doc_ids', (len(self.orig_doc_ids),), dtype=str_dt)
            }
            for q_id, query in tqdm(self.queries.items(), desc='Saving queries'):
                ds['queries'][q_id] = query
                ds['orig_q_ids'][q_id] = self.orig_q_ids[q_id]

            for doc_id, doc in tqdm(self.docs.items(), desc='Saving documents'):
                ds['docs'][doc_id] = doc
                ds['orig_doc_ids'][doc_id] = self.orig_doc_ids[doc_id]

    def save_qrels(self, dest: Path):
        """Save the QRels as a tab-separated file to be used with TREC-eval.
        Positive values indicate relevant documents. Zero or negative values indicate irrelevant documents.

        Args:
            dest (Path): The file to create
        """
        with open(dest, 'w', encoding='utf-8', newline='') as fp:
            writer = csv.writer(fp, delimiter='\t')
            for q_id in self.qrels:
                for doc_id, rel in self.qrels[q_id].items():
                    orig_q_id = self.orig_q_ids[q_id]
                    orig_doc_id = self.orig_doc_ids[doc_id]
                    writer.writerow([orig_q_id, 0, orig_doc_id, rel])

    def save(self, directory: Path,
             num_neg_point: Optional[int] = None,
             num_neg_pair: Optional[int] = None, query_limit_pair: Optional[int] = None):
        """Save the collection, QRels and all folds of trainingsets, validationset and testset.

        Args:
            directory (Path): Where to save the files
            num_neg_point (Optional[int], optional): Number of negatives per positive (pointwise training). Defaults to None.
            num_neg_pair (Optional[int], optional): Number of negatives per positive (pairwise training). Defaults to None.
            query_limit_pair (Optional[int], optional): Maximum number of training examples per query (pairwise training). Defaults to None.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.save_collection(directory / 'data.h5')
        self.save_qrels(directory / 'qrels.tsv')
        for fold in range(len(self.folds)):
            fold_dir = directory / f'fold_{fold}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            if num_neg_point is not None:
                self.get_pointwise_trainingset(fold, num_neg_point).save(fold_dir / 'train_pointwise.h5')
            if num_neg_pair is not None and query_limit_pair is not None:
                self.get_pairwise_trainingset(fold, num_neg_pair, query_limit_pair).save(fold_dir / 'train_pairwise.h5')
            self.get_valset(fold).save(fold_dir / 'val.h5')
            self.get_testset(fold).save(fold_dir / 'test.h5')


class ParsableDataset(Dataset, abc.ABC):
    def __init__(self, args: argparse.Namespace):
        """Abstract base class for datasets that are parsed from files.

        Args:
            args (argparse.Namespace): Namespace that contains the arguments
        """
        self.directory = Path(args.DIRECTORY)
        queries = self.get_queries()
        docs = self.get_docs()
        qrels = self.get_qrels()
        pools = self.get_pools()
        super().__init__(queries, docs, qrels, pools)
        for f in self.get_folds():
            self.add_fold(*f)

    @abc.abstractmethod
    def get_queries(self) -> Dict[str, str]:
        """Return all queries.

        Returns:
            Dict[str, str]: Query IDs mapped to queries
        """
        pass

    @abc.abstractmethod
    def get_docs(self) -> Dict[str, str]:
        """Return all documents

        Returns:
            Dict[str, str]: Document IDs mapped to documents
        """
        pass

    @abc.abstractmethod
    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all query relevances.

        Returns:
            Dict[str, Dict[str, int]]: Query IDs mapped to document IDs mapped to relevance
        """
        pass

    @abc.abstractmethod
    def get_pools(self) -> Dict[str, Set[str]]:
        """Return all pools.

        Returns:
            Dict[str, Set[str]]: Query IDs mapped to top retrieved documents
        """
        pass

    @abc.abstractmethod
    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        """Return all folds.

        Returns:
            Iterable[Tuple[Set[str], Set[str], Set[str]]]: Folds of train, validation and test query IDs
        """
        pass

    @staticmethod
    def add_subparser(subparsers: argparse._SubParsersAction, name: str):
        """Add a dataset-specific subparser with all required arguments.

        Args:
            subparsers (argparse._SubParsersAction): Subparsers to add a parser to
            name (str): Parser name
        """
        sp = subparsers.add_parser(name)
        sp.add_argument('DIRECTORY', help='Dataset directory containing all files')
