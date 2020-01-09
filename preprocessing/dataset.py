import os
import abc
import random

from tqdm import tqdm


class Dataset(abc.ABC):
    """Abstract base class for datasets.

    Arguments:
        args {argparse.Namespace} -- The command line arguments
    """

    def __init__(self, args):
        self.args = args
        self.queries, self.docs, self.qrels, self.train_q_ids, self.dev_set, self.test_set = \
            self._read_dataset()

    @abc.abstractmethod
    def _read_dataset(self):
        """Read all dataset files.

        Returns:
            tuple[dict[int, str], dict[int, str], dict[int, set[int]], set[int],
                  dict[int, tuple[int, int]], dict[int, tuple[int, int]]] -- A tuple containing
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
        return Trainset(self.queries, self.docs, self.train_q_ids, self.qrels,
                        self.args.num_neg_examples)

    @property
    def devset(self):
        """A devset iterator.

        Returns:
            Testset -- The devset
        """
        return Testset(self.queries, self.docs, self.dev_set)

    @property
    def testset(self):
        """A testset iterator.

        Returns:
            Testset -- The testset
        """
        return Testset(self.queries, self.docs, self.test_set)

    def transform_queries(self, f):
        """Apply a function to all queries.

        Arguments:
            f {function} -- The function to apply
        """
        for q_id in tqdm(self.queries):
            self.queries[q_id] = f(self.queries[q_id])

    def transform_docs(self, f):
        """Apply a function to all documents.

        Arguments:
            f {function} -- The function to apply
        """
        for doc_id in tqdm(self.docs):
            self.docs[doc_id] = f(self.docs[doc_id])

    @staticmethod
    @abc.abstractmethod
    def add_subparser(subparsers):
        """Add a dataset-specific subparser with all required arguments.

        Arguments:
            subparsers {argparse._SubParsersAction} -- The subparsers object to add the subparser to
        """
        pass


class Trainset(object):
    """A trainset iterator.

    Arguments:
        queries {dict[int, str]} -- The queries
        docs {dict[int, str]} -- The documents
        train_q_ids {list[int]} -- The query IDs in the trainset
        train_qrels {dict[int, list[int]]} -- Relevant documents for each query
        num_neg_examples {int} -- Number of negative examples for each positive one

    Yields:
        tuple[str, str, list[str]] -- A tuple containing
            * a query
            * a relevant document
            * a list of irrelevant documents
    """

    def __init__(self, queries, docs, train_q_ids, train_qrels, num_neg_examples):
        self.queries = queries
        self.docs = docs
        self.train_q_ids = train_q_ids
        self.train_qrels = train_qrels
        self.num_neg_examples = num_neg_examples

        # enumerate all positive (query, document) pairs
        self.pos_pairs = []
        for q_id in train_q_ids:
            # empty queries or documents will cause errors
            if len(queries[q_id]) == 0:
                continue
            for doc_id in train_qrels[q_id]:
                if len(docs[doc_id]) == 0:
                    continue
                self.pos_pairs.append((q_id, doc_id))

        # a list of all doc ids to sample negatives from
        self.neg_sample_doc_ids = set()
        for doc_id, doc in docs.items():
            if len(doc) > 0:
                self.neg_sample_doc_ids.add(doc_id)
        self.neg_sample_doc_ids = list(self.neg_sample_doc_ids)

    def _sample_negatives(self, q_id):
        """Sample a number of negative/irrelevant documents for a query.

        Arguments:
            q_id {int} -- A query ID

        Returns:
            list[int] -- A list of irrelevant document IDs
        """

        rels = self.train_qrels[q_id]
        sampled_docs = set()

        for _ in range(self.num_neg_examples):
            sampled_id = random.choice(self.neg_sample_doc_ids)
            # redo sampling if relevant to the query or if we have already sampled it
            while sampled_id in rels or sampled_id in sampled_docs:
                sampled_id = random.choice(self.neg_sample_doc_ids)
            sampled_docs.add(sampled_id)

        return sampled_docs

    def _get_train_examples(self):
        """Yield all training examples.

        Yields:
            tuple[str, str, list[str]] -- A tuple containing
                * a query
                * a relevant document
                * a list of irrelevant documents
        """
        for q_id, pos_doc_id in self.pos_pairs:
            neg_docs = [self.docs[neg_doc_id] for neg_doc_id in self._sample_negatives(q_id)]
            yield self.queries[q_id], self.docs[pos_doc_id], neg_docs

    def __len__(self):
        return len(self.pos_pairs)

    def __iter__(self):
        yield from self._get_train_examples()


class Testset(object):
    """A dev-/testset iterator.

    Arguments:
        queries {dict[int, str]} -- The queries
        docs {dict[int, str]} -- The documents
        test_set {dict[int, list[tuple[int, int]]]} -- A map of query IDs to document IDs and labels

    Yields:
        tuple[int, str, str, int] -- A query ID, a query, a document and a binary label
    """

    def __init__(self, queries, docs, test_set):
        self.queries = queries
        self.docs = docs
        self.test_set = test_set

    def _get_test_examples(self):
        """Yield all test examples.

        Yields:
            tuple[int, str, str, int] -- A query ID, a query, a document and a binary label
        """
        for q_id, doc_ids in self.test_set.items():
            # empty/nonexistent queries or documents will cause errors
            if len(self.queries[q_id]) == 0:
                continue
            for doc_id, label in doc_ids:
                if len(self.docs[doc_id]) > 0:
                    yield q_id, self.queries[q_id], self.docs[doc_id], label

    def __len__(self):
        return sum(map(len, self.test_set.values()))

    def __iter__(self):
        yield from self._get_test_examples()
