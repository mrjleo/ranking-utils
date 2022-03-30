import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from ranking_utils.datasets import ParsableDataset


class InsuranceQAV2(ParsableDataset):
    """InsuranceQA (v2) dataset class."""

    def __init__(self, root_dir: Path, pool_size: int = 100):
        """Constructor.

        Args:
            root_dir (Path): Directory that contains all dataset files.
            pool_size (int): Number of documents per query (100, 500, 1000, 1500). Defaults to 100.
        """
        assert pool_size in (100, 500, 1000, 1500)
        self.pool_size = pool_size
        super().__init__(root_dir)

    def prepare_data(self) -> None:
        vocab_file = self.root_dir / "vocabulary"
        vocab = {}
        with open(vocab_file, encoding="utf-8", newline="") as fp:
            for idx, word in csv.reader(fp, delimiter="\t", quotechar=None):
                vocab[idx] = word

        def _decode(idx_list):
            return " ".join(map(vocab.get, idx_list))

        l2a_file = self.root_dir / "InsuranceQA.label2answer.token.encoded.gz"
        self.docs = {}
        with gzip.open(l2a_file) as fp:
            for line in fp:
                doc_id, doc_idxs = line.decode("utf-8").split("\t")
                self.docs[doc_id] = _decode(doc_idxs.split())

        # read all qrels and top documents
        files = [
            self.root_dir
            / f"InsuranceQA.question.anslabel.token.{self.pool_size}.pool.solr.train.encoded.gz",
            self.root_dir
            / f"InsuranceQA.question.anslabel.token.{self.pool_size}.pool.solr.valid.encoded.gz",
            self.root_dir
            / f"InsuranceQA.question.anslabel.token.{self.pool_size}.pool.solr.test.encoded.gz",
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
