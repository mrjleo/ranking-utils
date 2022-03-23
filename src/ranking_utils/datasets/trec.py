import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from ranking_utils.datasets import ParsableDataset


def read_qrels_trec(fname: Path) -> Dict[str, Dict[str, int]]:
    """Read query relevances, TREC format.

    Args:
        fname (Path): Query relevances file.

    Returns:
        Dict[str, Dict[str, int]]: Query IDs mapped to tuples of document ID and relevance.
    """
    qrels = defaultdict(dict)
    with open(fname, encoding="utf-8") as fp:
        for line in fp:
            row = line.split()
            q_id = row[0]
            doc_id = row[2]
            rel = int(row[3])
            qrels[q_id][doc_id] = rel
    return qrels


def read_top_trec(fname: Path) -> Dict[int, Set[str]]:
    """Read the top document IDs for each query, TREC format.

    Args:
        fname (Path): File to read from.

    Returns:
        Dict[str, Set[str]]: Query IDs mapped to top documents.
    """
    top = defaultdict(set)
    with open(fname, encoding="utf-8") as fp:
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

    def get_queries(self) -> Dict[str, str]:
        queries = {}
        with open(self.root_dir / "queries.tsv", encoding="utf-8", newline="") as fp:
            for q_id, query, _, _ in csv.reader(fp, delimiter="\t"):
                queries[q_id] = query
        return queries

    def get_docs(self) -> Dict[str, str]:
        docs = {}
        with open(self.root_dir / "documents.tsv", encoding="utf-8", newline="") as fp:
            for doc_id, doc in csv.reader(fp, delimiter="\t"):
                docs[doc_id] = doc
        return docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        return read_qrels_trec(self.root_dir / "qrels.tsv")

    def get_pools(self) -> Dict[str, Set[str]]:
        return read_top_trec(self.root_dir / "top.tsv")

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        folds = []
        folds_dir = self.root_dir / "folds"
        for fold_dir in sorted(list(folds_dir.iterdir())):
            with open(folds_dir / fold_dir / "train_ids.txt") as fp:
                train_ids = set([l.strip() for l in fp])
            with open(folds_dir / fold_dir / "val_ids.txt") as fp:
                val_ids = set([l.strip() for l in fp])
            with open(folds_dir / fold_dir / "test_ids.txt") as fp:
                test_ids = set([l.strip() for l in fp])
            folds.append((train_ids, val_ids, test_ids))
        return folds
