import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from ranking_utils.dataset import ParsableDataset
from ranking_utils.datasets.trec import read_qrels_trec, read_top_trec
from tqdm import tqdm

# some documents are longer than the default limit
csv.field_size_limit(sys.maxsize)


class TRECDL2019Passage(ParsableDataset):
    """TREC-DL 2019 passage ranking dataset class.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments.
    """

    def __init__(self, args: argparse.Namespace):
        self.directory = Path(args.DIRECTORY)
        self._read_all()
        super().__init__(args)

    def _read_all(self):
        """Read the dataset."""
        # read queries
        self.queries = {}
        for f_name, num_lines in [
            ("queries.train.tsv", 808731),
            ("queries.dev.tsv", 101093),
            ("msmarco-test2019-queries.tsv", 200),
        ]:
            f = self.directory / f_name
            print(f"reading {f}...")
            with open(f, encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp, delimiter="\t")
                for q_id, query in tqdm(reader, total=num_lines):
                    self.queries[q_id] = query

        # read documents
        self.docs = {}
        f = self.directory / "collection.tsv"
        print(f"reading {f}...")
        with open(f, encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp, delimiter="\t")
            for doc_id, doc in tqdm(reader, total=8841823):
                self.docs[doc_id] = doc

        # read qrels
        self.qrels = defaultdict(dict)
        q_ids = defaultdict(set)
        for f_name, num_lines in [
            ("qrels.train.tsv", 532761),
            ("qrels.dev.tsv", 59273),
        ]:
            f = self.directory / f_name
            print(f"reading {f}...")
            with open(f, encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp, delimiter="\t")
                for q_id, _, doc_id, rel in tqdm(reader, total=num_lines):
                    self.qrels[q_id][doc_id] = int(rel)
                    q_ids[f_name].add(q_id)

        # TREC qrels have a different format
        f = self.directory / "2019qrels-pass.txt"
        print(f"reading {f}...")
        with open(f, encoding="utf-8", newline="") as fp:
            for q_id, _, doc_id, rel in csv.reader(fp, delimiter=" "):
                # 1 is considered irrelevant
                self.qrels[q_id][doc_id] = int(rel) - 1
                q_ids["2019qrels-pass.txt"].add(q_id)

        # read top documents
        self.pools = defaultdict(set)
        for f_name, num_lines in [
            ("top1000.dev.tsv", 6668967),
            ("msmarco-passagetest2019-top1000.tsv", 189877),
            ("top1000.train.txt", 478016942),
        ]:
            f = self.directory / f_name
            print(f"reading {f}...")
            with open(f, encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp, delimiter="\t")
                for q_id, doc_id, _, _ in tqdm(reader, total=num_lines):
                    self.pools[q_id].add(doc_id)

        # some IDs have no pool or no query -- remove them
        all_ids = set(self.pools.keys()) & set(self.queries.keys())
        self.train_ids = q_ids["qrels.train.tsv"] & all_ids
        self.val_ids = q_ids["qrels.dev.tsv"] & all_ids
        self.test_ids = q_ids["2019qrels-pass.txt"] & all_ids

    def get_queries(self) -> Dict[str, str]:
        """Return all queries.

        Returns:
            Dict[str, str]: Query IDs mapped to queries.
        """
        return self.queries

    def get_docs(self) -> Dict[str, str]:
        """Return all documents.

        Returns:
            Dict[str, str]: Document IDs mapped to documents.
        """
        return self.docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all query relevances.

        Returns:
            Dict[str, Dict[str, int]]: Query IDs mapped to document IDs mapped to relevance.
        """
        return self.qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        """Return all pools.

        Returns:
            Dict[str, Set[str]]: Query IDs mapped to top retrieved documents.
        """
        return self.pools

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        """Return all folds.

        Returns:
            Iterable[Tuple[Set[str], Set[str], Set[str]]]: Folds of training, validation and test query IDs.
        """
        return [(self.train_ids, self.val_ids, self.test_ids)]


class TRECDL2019Document(ParsableDataset):
    """TREC-DL 2019 document ranking dataset class.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments.
    """

    def __init__(self, args: argparse.Namespace):
        self.directory = Path(args.DIRECTORY)
        self._read_queries()
        super().__init__(args)

    def _read_queries(self):
        """Read the queries and split."""

        def _read_queries(fname):
            result = {}
            with open(fname, encoding="utf-8", newline="") as fp:
                for q_id, query in csv.reader(fp, delimiter="\t"):
                    result[q_id] = query
            return result

        train_queries = _read_queries(self.directory / "msmarco-doctrain-queries.tsv")
        dev_queries = _read_queries(self.directory / "msmarco-docdev-queries.tsv")
        test_queries = _read_queries(self.directory / "msmarco-test2019-queries.tsv")

        self.queries = {}
        self.queries.update(train_queries)
        self.queries.update(dev_queries)
        self.queries.update(test_queries)

        self.train_ids = set(train_queries.keys())
        self.test_ids = set(test_queries.keys())
        self.val_ids = set(dev_queries.keys())

    def get_queries(self) -> Dict[str, str]:
        """Return all queries.

        Returns:
            Dict[str, str]: Query IDs mapped to queries.
        """
        return self.queries

    def get_docs(self) -> Dict[str, str]:
        """Return all documents.

        Returns:
            Dict[str, str]: Document IDs mapped to documents.
        """
        docs = {}
        with open(
            self.directory / "msmarco-docs.tsv", encoding="utf-8", newline=""
        ) as fp:
            for doc_id, _, title, body in tqdm(
                csv.reader(fp, delimiter="\t"), total=3213835
            ):
                doc = title + ". " + body
                docs[doc_id] = doc
        return docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all query relevances.

        Returns:
            Dict[str, Dict[str, int]]: Query IDs mapped to document IDs mapped to relevance.
        """
        qrels = {}
        qrels.update(read_qrels_trec(self.directory / "msmarco-doctrain-qrels.tsv"))
        qrels.update(read_qrels_trec(self.directory / "msmarco-docdev-qrels.tsv"))
        qrels.update(read_qrels_trec(self.directory / "2019qrels-docs.txt"))
        return qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        """Return all pools.

        Returns:
            Dict[str, Set[str]]: Query IDs mapped to top retrieved documents.
        """
        top = {}
        top.update(read_top_trec(self.directory / "msmarco-doctrain-top100"))
        top.update(read_top_trec(self.directory / "msmarco-docdev-top100"))
        top.update(read_top_trec(self.directory / "msmarco-doctest2019-top100"))
        return top

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        """Return all folds.

        Returns:
            Iterable[Tuple[Set[str], Set[str], Set[str]]]: Folds of training, validation and test query IDs.
        """
        return [(self.train_ids, self.val_ids, self.test_ids)]
