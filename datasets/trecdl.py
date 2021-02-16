import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Set, Tuple

from tqdm import tqdm

from ranking_utils.dataset import ParsableDataset


class TRECDL2019Passage(ParsableDataset):
    """TREC-DL 2019 passage ranking dataset.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments
    """
    def __init__(self, args: argparse.Namespace):
        self.directory = Path(args.DIRECTORY)
        self._read_all()
        super().__init__(args)

    def _read_all(self):
        """Read the dataset."""
        # read queries
        self.queries = {}
        for f_name, num_lines in [('queries.train.tsv', 808731),
                                  ('queries.dev.tsv', 101093),
                                  ('msmarco-test2019-queries.tsv', 200)]:
            f = self.directory / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, query in tqdm(reader, total=num_lines):
                    self.queries[q_id] = query

        # read documents
        self.docs = {}
        f = self.directory / 'collection.tsv'
        print(f'reading {f}...')
        with open(f, encoding='utf-8') as fp:
            reader = csv.reader(fp, delimiter='\t')
            for doc_id, doc in tqdm(reader, total=8841823):
                self.docs[doc_id] = doc

        # read qrels
        self.qrels = defaultdict(dict)
        q_ids = defaultdict(set)
        for f_name, num_lines in [('qrels.train.tsv', 532761), ('qrels.dev.tsv', 59273)]:
            f = self.directory / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, _, doc_id, rel in tqdm(reader, total=num_lines):
                    self.qrels[q_id][doc_id] = int(rel)
                    q_ids[f_name].add(q_id)

        # TREC qrels have a different format
        f = self.directory / '2019qrels-pass.txt'
        print(f'reading {f}...')
        with open(f, encoding='utf-8') as fp:
            for q_id, _, doc_id, rel in csv.reader(fp, delimiter=' '):
                # 1 is considered irrelevant
                self.qrels[q_id][doc_id] = int(rel) - 1
                q_ids['2019qrels-pass.txt'].add(q_id)

        # read top documents
        self.pools = defaultdict(set)
        for f_name, num_lines in [('top1000.train.txt', 478016942),
                                  ('top1000.dev.tsv', 6668967),
                                  ('msmarco-passagetest2019-top1000.tsv', 189877)]:
            f = self.directory / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, doc_id, _, _ in tqdm(reader, total=num_lines):
                    self.pools[q_id].add(doc_id)

        # some IDs have no pool or no query -- remove them
        all_ids = set(self.pools.keys()) & set(self.queries.keys())
        self.train_ids = q_ids['qrels.train.tsv'] & all_ids
        self.val_ids = q_ids['qrels.dev.tsv'] & all_ids
        self.test_ids = q_ids['2019qrels-pass.txt'] & all_ids

    def get_queries(self) -> Dict[str, str]:
        """Return all queries.

        Returns:
            Dict[str, str]: Query IDs mapped to queries
        """
        return self.queries

    def get_docs(self) -> Dict[str, str]:
        """Return all documents

        Returns:
            Dict[str, str]: Document IDs mapped to documents
        """
        return self.docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all query relevances.

        Returns:
            Dict[str, Dict[str, int]]: Query IDs mapped to document IDs mapped to relevance
        """
        return self.qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        """Return all pools.

        Returns:
            Dict[str, Set[str]]: Query IDs mapped to top retrieved documents
        """
        return self.pools

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        """Return all folds.

        Returns:
            Iterable[Tuple[Set[str], Set[str], Set[str]]]: Folds of train, validation and test query IDs
        """
        return [(self.train_ids, self.val_ids, self.test_ids)]