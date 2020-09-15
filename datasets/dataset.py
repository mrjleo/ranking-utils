import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Sequence, Set, Tuple, List, Iterable

import h5py
from tqdm import tqdm


class Trainingset(object):
    """A trainingset iterator.

    Arguments:
        doc_ids (Sequence[int]): All documents IDs
        train_set (Dict[int, List[Tuple[int, int]]]): Train query IDs mapped to document IDs and labels
        num_negatives (int): Number of negative examples for each positive one

    Yields:
        Tuple[int, int, int]: Query ID, positive document ID, negative document ID
    """
    def __init__(self, doc_ids: Sequence[int], train_set: Dict[int, List[Tuple[int, int]]], num_negatives: int):
        self.doc_ids = doc_ids
        self.num_negatives = num_negatives

        self.train_positives, self.train_negatives = defaultdict(set), defaultdict(set)
        for q_id, items in train_set.items():
            for doc_id, label in items:
                if label == 1:
                    self.train_positives[q_id].add(doc_id)
                else:
                    self.train_negatives[q_id].add(doc_id)

    def _sample_negatives(self, q_id: int) -> Set[int]:
        """Sample negative documents for a query.

        Args:
            q_id (int): Query ID

        Returns:
            Set[int]: Negative document IDs
        """
        # sample from the known negatives if there are enough
        if self.num_negatives <= len(self.train_negatives[q_id]):
            return random.sample(self.train_negatives[q_id], self.num_negatives)

        # otherwise, take all known negatives and sample the rest randomly from all docs
        sample = set(self.train_negatives[q_id])
        while len(sample) < self.num_negatives:
            doc_id = random.choice(self.doc_ids)
            if doc_id not in self.train_positives[q_id]:
                sample.add(doc_id)
        return sample

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            int: The number of training pairs
        """
        return sum(map(len, self.train_positives.values())) * self.num_negatives

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
        """Yield all training examples.

        Yields:
            Tuple[int, int, int]: Query ID, positive document ID, negative document ID
        """
        for q_id in self.train_positives:
            for pos_doc_id in self.train_positives[q_id]:
                for neg_doc_id in self._sample_negatives(q_id):
                    yield q_id, pos_doc_id, neg_doc_id

    def save(self, dest: Path):
        """Save the trainingset for pairwise training.

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
            for i, (q_id, pos_doc_id, neg_doc_id) in enumerate(tqdm(self, desc='Saving trainset')):
                ds['q_ids'][i] = q_id
                ds['pos_doc_ids'][i] = pos_doc_id
                ds['neg_doc_ids'][i] = neg_doc_id


class Testset(object):
    """A Testset iterator.

    Arguments:
        test_set (Dict[int, List[Tuple[int, int]]]): Test query IDs mapped to document IDs and labels

    Yields:
        Tuple[int, int, int]: Query ID, document ID, label
    """
    def __init__(self, test_set: Dict[int, List[Tuple[int, int]]]):
        self.test_set = test_set

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            int: Number of test items
        """
        return sum(map(len, self.test_set.values()))

    def __iter__(self) -> Iterable[Tuple[int, int, int]]:
        """Yield all test items.

        Yields:
            Tuple[int, int, int]: Query ID, document ID, label
        """
        for q_id, doc_ids in self.test_set.items():
            for doc_id, label in doc_ids:
                yield q_id, doc_id, label

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
                'queries': fp.create_dataset('queries', (num_items,), dtype='int32'),
                'labels': fp.create_dataset('labels', (num_items,), dtype='int32'),
                'offsets': fp.create_dataset('offsets', (len(self.test_set),), dtype='int32')
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
            assert i_offsets == len(self.test_set)


class Dataset(object):
    """Dataset class that provides iterators over train-, val- and testset. for the trainingset,
    positive documents are taken from `qrels` and negative ones are sampled from `pools`.

    Similarly, validation- and testset contain all documents (corresponding to the query IDs in the set)
    from `qrels` and `pools`, which are to be re-ranked.

    Query and document IDs are converted to integer internally. Original IDs can be restored using
    `orig_q_ids` and `orig_doc_ids`.

    Args:
        queries (Dict[str, str]): Query IDs mapped to queries
        docs (Dict[str, str]): Document IDs mapped to documents
        qrels (Dict[str, Set[str]]): Query IDs mapped to relevant documents
        pools (Dict[str, Set[str]]): Query IDs mapped to top-k retrieved documents
        train_ids (Set[str]): Trainset query IDs
        val_ids (Set[str]): Validationset query IDs
        test_ids (Set[str]): Testset query IDs
        num_negatives (int): Number of negative training examples per positive
    """
    def __init__(self,
                 queries: Dict[str, str], docs: Dict[str, str],
                 qrels: Dict[str, Set[str]], pools: Dict[str, Set[str]],
                 train_ids: Set[str], val_ids: Set[str], test_ids: Set[str],
                 num_negatives: int):
        # make sure no IDs are in any two sets
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

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

        # use integer IDs for everything
        self.qrels = {}
        for orig_q_id, orig_doc_ids in qrels.items():
            int_q_id = self.int_q_ids[orig_q_id]
            int_doc_ids = {self.int_doc_ids[orig_doc_id] for orig_doc_id in orig_doc_ids if orig_doc_id in self.int_doc_ids}
            self.qrels[int_q_id] = int_doc_ids

        self.pools = {}
        for orig_q_id, orig_doc_ids in pools.items():
            int_q_id = self.int_q_ids[orig_q_id]
            int_doc_ids = {self.int_doc_ids[orig_doc_id] for orig_doc_id in orig_doc_ids if orig_doc_id in self.int_doc_ids}
            self.pools[int_q_id] = int_doc_ids

        self.train_set = self._create_set(map(self.int_q_ids.get, train_ids))
        self.val_set = self._create_set(map(self.int_q_ids.get, val_ids))
        self.test_set = self._create_set(map(self.int_q_ids.get, test_ids))
        self.num_negatives = num_negatives

    def _create_set(self, q_ids: Sequence[int]) -> Dict[int, List[Tuple[int, int]]]:
        """Create a set of documents for the query IDs. For each query, the set contains its pool and all relevant documents.
        Empty queries and documents will be ignored.

        Args:
            q_ids (Sequence[int]): Query IDs

        Returns:
            Dict[int, List[Tuple[int, int]]]: Query IDs mapped to document IDs and labels
        """
        result = defaultdict(list)
        for q_id in q_ids:
            if len(self.queries.get(q_id, '').strip()) == 0:
                continue

            for doc_id in self.pools.get(q_id, set()) | self.qrels.get(q_id, set()):
                if len(self.docs.get(doc_id, '').strip()) > 0:
                    label = 1 if doc_id in self.qrels.get(q_id, set()) else 0
                    result[q_id].append((doc_id, label))
        return result

    @property
    def trainset(self) -> Trainingset:
        """Trainingset iterator.

        Returns:
            Trainset: The trainingset
        """
        return Trainingset(list(self.docs.keys()), self.train_set, self.num_negatives)

    @property
    def valset(self) -> Testset:
        """Validationset iterator

        Returns:
            Testset: The validationset
        """
        return Testset(self.val_set)

    @property
    def testset(self) -> Testset:
        """Testset iterator

        Returns:
            Testset: The testset
        """
        return Testset(self.test_set)

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

    def save(self, directory: Path):
        """Save the collection, trainingset, validationset and testset.

        Args:
            directory (Path): Where to save the files
        """
        self.save_collection(directory / 'data.h5')
        self.trainset.save(directory / 'train.h5')
        self.valset.save(directory / 'val.h5')
        self.testset.save(directory / 'test.h5')
