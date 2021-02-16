import csv
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Set, Tuple

from ranking_utils.dataset import ParsableDataset


class ANTIQUE(ParsableDataset):
    """ANTIQUE dataset class."""
    def get_queries(self) -> Dict[str, str]:
        """Return all queries.

        Returns:
            Dict[str, str]: Query IDs mapped to queries
        """
        queries = {}
        for f_name in ['antique-train-queries.txt', 'antique-test-queries.txt']:
            f = self.directory / f_name
            with open(f, encoding='utf-8') as fp:
                queries.update({q_id: query for q_id, query in csv.reader(fp, delimiter='\t')})
        return queries

    def get_docs(self) -> Dict[str, str]:
        """Return all documents

        Returns:
            Dict[str, str]: Document IDs mapped to documents
        """
        doc_file = self.directory / 'antique-collection.txt'
        with open(doc_file, encoding='utf-8') as fp:
            return {doc_id: doc for doc_id, doc in csv.reader(fp, delimiter='\t', quotechar=None)}

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all query relevances.

        Returns:
            Dict[str, Dict[str, int]]: Query IDs mapped to document IDs mapped to relevance
        """
        qrels = defaultdict(dict)
        for f_name in ['antique-train.qrel', 'antique-test.qrel']:
            f = self.directory / f_name
            with open(f, encoding='utf-8') as fp:
                for line in fp:
                    q_id, _, doc_id, rel = line.split()

                    # the authors recommend treating rel > 2 as positive
                    qrels[q_id][doc_id] = int(rel) - 2
        return qrels

    def get_pools(self) -> Dict[str, Set[str]]:
        """Return all pools.

        Returns:
            Dict[str, Set[str]]: Query IDs mapped to top retrieved documents
        """
        split_file = Path(__file__).parent.absolute() / 'splits' / 'antique_split.pkl'
        with open(split_file, 'rb') as fp:
            pools, _ = pickle.load(fp)
        
        # for the testset, we create the pools from the qrels
        with open(self.directory / 'antique-test.qrel', encoding='utf-8') as fp:
            for line in fp:
                q_id, _, doc_id, _ = line.split()
                pools[q_id].add(doc_id)
        return pools

    def get_folds(self) -> Iterable[Tuple[Set[str], Set[str], Set[str]]]:
        """Return all folds.

        Returns:
            Iterable[Tuple[Set[str], Set[str], Set[str]]]: Folds of train, validation and test query IDs
        """
        split_file = Path(__file__).parent.absolute() / 'splits' / 'antique_split.pkl'
        with open(split_file, 'rb') as fp:
            _, val_ids = pickle.load(fp)
        
        train_ids, test_ids = set(), set()
    
        with open(self.directory / 'antique-train.qrel', encoding='utf-8') as fp:
            for line in fp:
                q_id, _, _, _ = line.split()
                if q_id not in val_ids:
                    train_ids.add(q_id)
        
        with open(self.directory / 'antique-test.qrel', encoding='utf-8') as fp:
            for line in fp:
                q_id, _, _, _ = line.split()
                test_ids.add(q_id)

        return [(train_ids, val_ids, test_ids)]
