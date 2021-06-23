import csv
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Set, Tuple

from ranking_utils.dataset import ParsableDataset


class FiQA(ParsableDataset):
    """FiQA dataset class.
    """
    def get_queries(self) -> Dict[str, str]:
        """Return all queries.

        Returns:
            Dict[str, str]: Query IDs mapped to queries
        """
        queries = {}
        with open(self.directory / 'FiQA_train_question_final.tsv', encoding='utf-8', newline='') as fp:
            # skip header
            next(fp)
            for _, q_id, question, _ in csv.reader(fp, delimiter='\t'):
                queries[q_id] = question
        return queries

    def get_docs(self) -> Dict[str, str]:
        """Return all documents.

        Returns:
            Dict[str, str]: Document IDs mapped to documents
        """
        docs = {}
        with open(self.directory / 'FiQA_train_doc_final.tsv', encoding='utf-8', newline='') as fp:
            # skip header
            next(fp)
            for _, doc_id, doc, _ in csv.reader(fp, delimiter='\t'):
                docs[doc_id] = doc
        return docs

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all query relevances.

        Returns:
            Dict[str, Dict[str, int]]: Query IDs mapped to document IDs mapped to relevance
        """
        qrels = defaultdict(dict)
        with open(self.directory / 'FiQA_train_question_doc_final.tsv', encoding='utf-8', newline='') as fp:
            # skip header
            next(fp)
            for _, q_id, doc_id in csv.reader(fp, delimiter='\t'):
                qrels[q_id][doc_id] = 1
        return qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        """Return all pools.

        Returns:
            Dict[str, Set[str]]: Query IDs mapped to top retrieved documents
        """
        split_file = Path(__file__).parent.absolute() / 'splits' / 'fiqa_split.pkl'
        with open(split_file, 'rb') as fp:
            pools, _, _ = pickle.load(fp)
        return pools

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        """Return all folds.

        Returns:
            Iterable[Tuple[Set[str], Set[str], Set[str]]]: Folds of train, validation and test query IDs
        """
        split_file = Path(__file__).parent.absolute() / 'splits' / 'fiqa_split.pkl'
        with open(split_file, 'rb') as fp:
            _, val_ids, test_ids = pickle.load(fp)

        train_ids = set()
        with open(self.directory / 'FiQA_train_question_final.tsv', encoding='utf-8', newline='') as fp:
            # skip header
            next(fp)
            for _, q_id, _, _ in csv.reader(fp, delimiter='\t'):
                if q_id not in val_ids | test_ids:
                    train_ids.add(q_id)

        return [(train_ids, val_ids, test_ids)]
        
