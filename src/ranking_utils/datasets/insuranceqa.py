import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from ranking_utils.datasets import ParsableDataset


class InsuranceQA(ParsableDataset):
    """InsuranceQA dataset class.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments
    """

    def __init__(self, args: argparse.Namespace):
        self.directory = Path(args.DIRECTORY)
        self.examples_per_query = args.examples_per_query
        self._read_all()
        super().__init__(args)

    def _read_all(self):
        """Read the dataset."""
        vocab_file = self.directory / "vocabulary"
        vocab = {}
        with open(vocab_file, encoding="utf-8", newline="") as fp:
            for idx, word in csv.reader(fp, delimiter="\t", quotechar=None):
                vocab[idx] = word

        def _decode(idx_list):
            return " ".join(map(vocab.get, idx_list))

        l2a_file = self.directory / "InsuranceQA.label2answer.token.encoded.gz"
        self.docs = {}
        with gzip.open(l2a_file) as fp:
            for line in fp:
                doc_id, doc_idxs = line.decode("utf-8").split("\t")
                self.docs[doc_id] = _decode(doc_idxs.split())

        # read all qrels and top documents
        files = [
            self.directory
            / f"InsuranceQA.question.anslabel.token.{self.examples_per_query}.pool.solr.train.encoded.gz",
            self.directory
            / f"InsuranceQA.question.anslabel.token.{self.examples_per_query}.pool.solr.valid.encoded.gz",
            self.directory
            / f"InsuranceQA.question.anslabel.token.{self.examples_per_query}.pool.solr.test.encoded.gz",
        ]
        sets = [set(), set(), set()]
        prefixes = ["train", "val", "test"]

        self.queries, self.qrels, self.pools = {}, defaultdict(dict), defaultdict(set)
        for f, ids, prefix in zip(files, sets, prefixes):
            with gzip.open(f) as fp:
                for i, line in enumerate(fp):
                    # we need to make the query IDs unique
                    q_id = f"{prefix}_{i}"
                    _, q_idxs, gt, pool = line.decode("utf-8").split("\t")
                    self.queries[q_id] = _decode(q_idxs.split())

                    ids.add(q_id)
                    for doc_id in gt.split():
                        self.qrels[q_id][doc_id] = 1
                    for doc_id in pool.split():
                        self.pools[q_id].add(doc_id)

        self.train_ids, self.val_ids, self.test_ids = sets

    def get_queries(self) -> Dict[str, str]:
        return self.queries

    def get_docs(self) -> Dict[str, str]:
        return self.docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        return self.qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        return self.pools

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        return [(self.train_ids, self.val_ids, self.test_ids)]

    @staticmethod
    def add_subparser(subparsers: argparse._SubParsersAction, name: str):
        sp = subparsers.add_parser(name)
        sp.add_argument("DIRECTORY", help="Dataset directory containing all files")
        sp.add_argument(
            "--examples_per_query",
            type=int,
            choices=[100, 500, 1000, 1500],
            default=500,
            help="How many examples per query in the dev- and testset",
        )
