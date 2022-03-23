import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from ranking_utils.datasets import ParsableDataset
from ranking_utils.datasets.trec import read_qrels_trec, read_top_trec
from tqdm import tqdm

# some documents are longer than the default limit
csv.field_size_limit(sys.maxsize)


LOGGER = logging.getLogger(__name__)


class MSMARCOPassage(ParsableDataset):
    """MS MARCO (v1) passage ranking dataset class. Includes TREC-DL 2019 and 2020 test sets.
    In the passage dataset, a relevance of 1 is considered irrelevant. Thus, we subtract 1 for each
    relevance while reading the QRels.
    """

    def __init__(self, root_dir: Path):
        """Constructor.

        Args:
            root_dir (Path): Directory that contains all dataset files.
        """
        # read queries
        self.queries = {}
        for f_name, num_lines in [
            ("queries.train.tsv", 808731),
            ("queries.dev.tsv", 101093),
            ("msmarco-test2019-queries.tsv", 200),
            ("msmarco-test2020-queries.tsv", 200),
        ]:
            f = root_dir / f_name
            LOGGER.info(f"reading {f}")
            with open(f, encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp, delimiter="\t")
                for q_id, query in tqdm(reader, total=num_lines):
                    self.queries[q_id] = query

        # read documents
        self.docs = {}
        f = root_dir / "collection.tsv"
        LOGGER.info(f"reading {f}")
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
            f = root_dir / f_name
            LOGGER.info(f"reading {f}")
            with open(f, encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp, delimiter="\t")
                for q_id, _, doc_id, rel in tqdm(reader, total=num_lines):
                    self.qrels[q_id][doc_id] = int(rel)
                    q_ids[f_name].add(q_id)

        # TREC qrels have a different format
        for f_name in ["2019qrels-pass.txt", "2020qrels-pass.txt"]:
            f = root_dir / f_name
            LOGGER.info(f"reading {f}")
            with open(f, encoding="utf-8", newline="") as fp:
                for q_id, _, doc_id, rel in csv.reader(fp, delimiter=" "):
                    # rel=1 is considered irrelevant
                    self.qrels[q_id][doc_id] = int(rel) - 1
                    q_ids[f_name].add(q_id)

        # read top documents
        self.pools = defaultdict(set)
        for f_name, num_lines in [
            ("msmarco-passagetest2019-top1000.tsv", 189877),
            ("msmarco-passagetest2020-top1000.tsv", 190699),
            ("top1000.dev.tsv", 6668967),
            ("top1000.train.txt", 478016942),
        ]:
            f = root_dir / f_name
            LOGGER.info(f"reading {f}")
            with open(f, encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp, delimiter="\t")
                for q_id, doc_id, _, _ in tqdm(reader, total=num_lines):
                    self.pools[q_id].add(doc_id)

        # some IDs have no pool or no query -- remove them
        all_ids = set(self.pools.keys()) & set(self.queries.keys())
        self.train_ids = q_ids["qrels.train.tsv"] & all_ids
        self.val_ids = q_ids["qrels.dev.tsv"] & all_ids
        self.test_ids_2019 = q_ids["2019qrels-pass.txt"] & all_ids
        self.test_ids_2020 = q_ids["2020qrels-pass.txt"] & all_ids

        super().__init__(root_dir)

    def get_queries(self) -> Dict[str, str]:
        return self.queries

    def get_docs(self) -> Dict[str, str]:
        return self.docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        return self.qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        return self.pools

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        return [
            (self.train_ids, self.val_ids, self.test_ids_2019),
            (self.train_ids, self.val_ids, self.test_ids_2020),
        ]


class MSMARCODocument(ParsableDataset):
    """MS MARCO (v1) document ranking dataset class. Includes TREC-DL 2019 and 2020 test sets."""

    def __init__(self, root_dir: Path):
        def _read_queries(fname):
            result = {}
            LOGGER.info(f"reading {fname}")
            with open(fname, encoding="utf-8", newline="") as fp:
                for q_id, query in csv.reader(fp, delimiter="\t"):
                    result[q_id] = query
            return result

        train_queries = _read_queries(root_dir / "msmarco-doctrain-queries.tsv")
        dev_queries = _read_queries(root_dir / "msmarco-docdev-queries.tsv")
        test_queries_2019 = _read_queries(root_dir / "msmarco-test2019-queries.tsv")
        test_queries_2020 = _read_queries(root_dir / "msmarco-test2020-queries.tsv")

        self.queries = {}
        self.queries.update(train_queries)
        self.queries.update(dev_queries)
        self.queries.update(test_queries_2019)
        self.queries.update(test_queries_2020)

        self.train_ids = set(train_queries.keys())
        self.val_ids = set(dev_queries.keys())
        self.test_ids_2019 = set(test_queries_2019.keys())
        self.test_ids_2020 = set(test_queries_2020.keys())

        super().__init__(root_dir)

    def get_queries(self) -> Dict[str, str]:
        return self.queries

    def get_docs(self) -> Dict[str, str]:
        docs = {}
        fname = self.root_dir / "msmarco-docs.tsv"
        LOGGER.info(f"reading {fname}")
        with open(fname, encoding="utf-8", newline="") as fp:
            for doc_id, _, title, body in tqdm(
                csv.reader(fp, delimiter="\t"), total=3213835
            ):
                doc = title + ". " + body
                docs[doc_id] = doc
        return docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        qrels = {}
        qrels.update(read_qrels_trec(self.root_dir / "msmarco-doctrain-qrels.tsv"))
        qrels.update(read_qrels_trec(self.root_dir / "msmarco-docdev-qrels.tsv"))
        qrels.update(read_qrels_trec(self.root_dir / "2019qrels-docs.txt"))
        qrels.update(read_qrels_trec(self.root_dir / "2020qrels-docs.txt"))
        return qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        top = {}
        top.update(read_top_trec(self.root_dir / "msmarco-doctrain-top100"))
        top.update(read_top_trec(self.root_dir / "msmarco-docdev-top100"))
        top.update(read_top_trec(self.root_dir / "msmarco-doctest2019-top100"))
        top.update(read_top_trec(self.root_dir / "msmarco-doctest2020-top100"))
        return top

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        return [
            (self.train_ids, self.val_ids, self.test_ids_2019),
            (self.train_ids, self.val_ids, self.test_ids_2020),
        ]
