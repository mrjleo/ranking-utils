import os
import csv
import pickle
from collections import defaultdict

from tqdm import tqdm

from qa_utils.preprocessing.misc import count_lines
from qa_utils.preprocessing.dataset import Dataset


def get_int_id(doc_id):
    # the IDs are strings, we create unique IDs for each doc
    i1, i2 = doc_id.split('_')
    return int(i1) * 10000 + int(i2)


def read_queries(file_path):
    with open(file_path, encoding='utf-8') as fp:
        return {int(q_id): query for q_id, query in csv.reader(fp, delimiter='\t')}


def read_qrels(file_path):
    qrels = defaultdict(set)
    with open(file_path, encoding='utf-8') as fp:
        for line in fp:
            q_id, _, doc_id, rel = line.split()
            if int(rel) > 2:
                qrels[int(q_id)].add(get_int_id(doc_id))
    return qrels


class Antique(Dataset):
    """Antique dataset class."""
    def _read_dataset(self):
        """Read all dataset files.

        Returns:
            tuple[dict, dict, dict, set, dict, dict] -- a tuple containing:
                * a mapping of query IDs to queries
                * a mapping of document IDs to documents
                * a mapping of query IDs to relevant document IDs
                * a set of query IDs in the trainset
                * a mapping of dev query IDs to tuples of (document ID, label)
                * a mapping of test query IDs to tuples of (document ID, label)
        """
        doc_file = os.path.join(self.args.ANTIQUE_DIR, 'antique-collection.txt')
        with open(doc_file, encoding='utf-8') as fp:
            docs = {get_int_id(doc_id): doc for doc_id, doc in csv.reader(fp, delimiter='\t',
                                                                          quotechar=None)}
        
        train_queries_file = os.path.join(self.args.ANTIQUE_DIR, 'antique-train-queries.txt')
        train_queries = read_queries(train_queries_file)

        train_qrels_file = os.path.join(self.args.ANTIQUE_DIR, 'antique-train.qrel')
        train_qrels = read_qrels(train_qrels_file)

        test_queries_file = os.path.join(self.args.ANTIQUE_DIR, 'antique-test-queries.txt')
        test_queries = read_queries(test_queries_file)

        test_qrels_file = os.path.join(self.args.ANTIQUE_DIR, 'antique-test.qrel')
        test_qrels = read_qrels(test_qrels_file)

        queries = train_queries.copy()
        queries.update(test_queries)

        qrels = train_qrels.copy()
        qrels.update(test_qrels)

        print('reading {}...'.format(self.args.SPLIT_FILE))
        with open(self.args.SPLIT_FILE, 'rb') as fp:
            train_q_ids, dev_set, test_set = pickle.load(fp)

        return queries, docs, qrels, train_q_ids, dev_set, test_set

    @staticmethod
    def add_subparser(subparsers, name):
        """Add a dataset-specific subparser with all required arguments."""
        sp = subparsers.add_parser(name)
        sp.add_argument('ANTIQUE_DIR', help='Folder with all Antique files')
        sp.add_argument('SPLIT_FILE', help='File with train/dev/test split')
