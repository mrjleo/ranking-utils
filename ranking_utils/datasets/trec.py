import csv
import ctypes
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from ranking_utils.datasets import ParsableDataset

# some documents are longer than the default limit
max_long = 2 ** (8 * ctypes.sizeof(ctypes.c_long) - 1) - 1
csv.field_size_limit(max_long)

LOGGER = logging.getLogger(__name__)


def read_qrels_trec(f: Path) -> Dict[str, Dict[str, int]]:
    """Read query relevances, TREC format.

    Args:
        f (Path): Query relevances file.

    Returns:
        Dict[str, Dict[str, int]]: Query IDs mapped to tuples of document ID and relevance.
    """
    qrels = defaultdict(dict)
    with open(f, encoding="utf-8") as fp:
        for line in fp:
            row = line.split()
            q_id = row[0]
            doc_id = row[2]
            rel = int(row[3])
            qrels[q_id][doc_id] = rel
    return qrels


def read_top_trec(f: Path) -> Dict[int, Set[str]]:
    """Read the top document IDs for each query, TREC format.

    Args:
        f (Path): File to read from.

    Returns:
        Dict[str, Set[str]]: Query IDs mapped to top documents.
    """
    top = defaultdict(set)
    with open(f, encoding="utf-8") as fp:
        for line in fp:
            row = line.split()
            q_id = row[0]
            doc_id = row[2]
            top[q_id].add(doc_id)
    return top


class TREC(ParsableDataset):
    """Generic TREC ranking dataset class.

    The directory must contain the following file structure:
        * queries.tsv -- queries, TREC format (tab separated)
        * documents.tsv -- documents, TREC format (tab separated)
        * qrels.tsv -- QRels, TREC format (space or tab separated)
        * top.tsv -- Top retrieved documents for each query (to be re-ranked), TREC format (space or tab separated)
        * folds -- directory
            * fold_0 -- directory
                * train_ids.txt -- training query IDs, one per line
                * val_ids.txt -- validation query IDs, one per line
                * test_ids.txt -- test query IDs, one per line
            * fold_1
                * ...
            * ...
    """

    def required_files(self) -> List[Path]:
        return [
            Path("queries.tsv"),
            Path("documents.tsv"),
            Path("qrels.tsv"),
            Path("top.tsv"),
        ]

    def get_queries(self) -> Dict[str, str]:
        queries = {}
        f = self.root_dir / "queries.tsv"
        LOGGER.info(f"reading {f}")
        with open(f, encoding="utf-8", newline="") as fp:
            for q_id, query, _, _ in csv.reader(fp, delimiter="\t"):
                queries[q_id] = query
        return queries

    def get_docs(self) -> Dict[str, str]:
        docs = {}
        f = self.root_dir / "documents.tsv"
        LOGGER.info(f"reading {f}")
        with open(f, encoding="utf-8", newline="") as fp:
            for doc_id, doc in csv.reader(fp, delimiter="\t"):
                docs[doc_id] = doc
        return docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        f = self.root_dir / "qrels.tsv"
        LOGGER.info(f"reading {f}")
        return read_qrels_trec(f)

    def get_pools(self) -> Dict[str, Set[str]]:
        f = self.root_dir / "top.tsv"
        LOGGER.info(f"reading {f}")
        return read_top_trec(f)

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        folds = []
        folds_dir = self.root_dir / "folds"
        for fold_dir in sorted(list(folds_dir.iterdir())):
            LOGGER.info(f"reading {fold_dir}")
            with open(folds_dir / fold_dir / "train_ids.txt") as fp:
                train_ids = set([l.strip() for l in fp])
            with open(folds_dir / fold_dir / "val_ids.txt") as fp:
                val_ids = set([l.strip() for l in fp])
            with open(folds_dir / fold_dir / "test_ids.txt") as fp:
                test_ids = set([l.strip() for l in fp])
            folds.append((train_ids, val_ids, test_ids))
        return folds
