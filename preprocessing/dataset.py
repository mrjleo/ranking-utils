import os
import abc
import random


class Dataset(abc.ABC):
    """Abstract base class for datasets."""    
    def __init__(self, args):
        self.args = args
        self.queries, self.docs, self.qrels, train_q_ids, self.dev_set, self.test_set = self._read_dataset()
        self.train_queries = {q_id: self.queries[q_id] for q_id in train_q_ids}
        self.dev_queries = {q_id: self.queries[q_id] for q_id in self.dev_set}
        self.test_queries = {q_id: self.queries[q_id] for q_id in self.test_set}

    @abc.abstractmethod
    def _read_dataset(self):
        """Read all dataset files.

        Returns:
            tuple[dict, dict, dict, set, dict, dict] -- a tuple containing:
                * a mapping of query IDs to queries
                * a mapping of document IDs to documents
                * a mapping of query IDs to relevant document IDs
                * a set of query IDs in the trainset
                * a mapping of dev query IDs to tuples of (document ID, label)
                * a mapping of test query IDs to tuples of (document ID, label)
        """
        pass

    @property
    def trainset(self):
        """A trainset iterator.

        Returns:
            Trainset -- The trainset
        """
        return Trainset(self.train_queries, self.docs, self.qrels, self.args.num_neg_examples)

    @property
    def devset(self):
        """A devset iterator.

        Returns:
            Testset -- The devset
        """
        return Testset(self.dev_queries, self.docs, self.dev_set)

    @property
    def testset(self):
        """A testset iterator.

        Returns:
            Testset -- The testset
        """
        return Testset(self.test_queries, self.docs, self.test_set)

    @staticmethod
    @abc.abstractmethod
    def add_subparser(subparsers):
        """Add a dataset-specific subparser with all required arguments."""
        pass


class Trainset(object):   
    def __init__(self, train_queries, docs, train_qrels, num_neg_examples):
        self.train_queries = train_queries
        self.docs = docs
        self.train_qrels = train_qrels
        self.num_neg_examples = num_neg_examples

        # enumerate all positive (query, document) pairs
        self.pos_pairs = []
        for q_id in train_queries:
            # empty queries or documents will cause errors
            if len(train_queries.get(q_id, [])) == 0:
                continue
            for doc_id in train_qrels[q_id]:
                if len(docs.get(doc_id, [])) == 0:
                    continue
                self.pos_pairs.append((q_id, doc_id))

        # a list of all doc ids to sample negatives from
        self.neg_sample_doc_ids = set()
        for doc_id, doc in docs.items():
            if len(doc) > 0:
                self.neg_sample_doc_ids.add(doc_id)

    def _sample_negatives(self, q_id):
        """Sample a number of negative/irrelevant documents for a query.

        Arguments:
            q_id {int} -- A query ID

        Returns:
            list[int] -- A list of irrelevant document IDs
        """
        population = self.neg_sample_doc_ids.copy()
        # the IDs of the docs that are relevant for this query (we can't use these as negatives)
        for doc_id in self.train_qrels[q_id]:
            if doc_id in population:
                population.remove(doc_id)
        return random.sample(population, self.num_neg_examples)

    def _get_train_examples(self):
        """Yield all training examples.

        Yields:
            tuple[str, str, list] -- a tuple containing
                * a query
                * a relevant document
                * a list of irrelevant documents
        """
        for q_id, pos_doc_id in self.pos_pairs:
            neg_docs = [self.docs[neg_doc_id] for neg_doc_id in self._sample_negatives(q_id)]
            yield self.train_queries[q_id], self.docs[pos_doc_id], neg_docs

    def __len__(self):
        return len(self.pos_pairs)

    def __iter__(self):
        yield from self._get_train_examples()


class Testset(object):
    def __init__(self, test_queries, docs, test_set):
        self.test_queries = test_queries
        self.docs = docs
        self.test_set = test_set

    def _get_test_examples(self):
        """Yield all test examples.

        Yields:
            tuple[int, str, str, int] -- a query ID, a query, a document and a binary label
        """
        for q_id, doc_ids in self.test_set.items():
            # empty/nonexistent queries or documents will cause errors
            if len(self.test_queries.get(q_id, [])) == 0:
                continue
            for doc_id, label in doc_ids:
                if len(self.docs[doc_id]) > 0:
                    yield q_id, self.test_queries[q_id], self.docs[doc_id], label

    def __len__(self):
        return sum(map(len, self.test_set.values()))

    def __iter__(self):
        yield from self._get_test_examples()
