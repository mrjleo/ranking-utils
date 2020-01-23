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


def read_train_set(file_path):
    train_set = defaultdict(list)
    with open(file_path, encoding='utf-8') as fp:
        for line in fp:
            q_id, _, doc_id, rel = line.split()
            if int(rel) > 2:
                train_set[int(q_id)].append((get_int_id(doc_id), 1))
            else:
                train_set[int(q_id)].append((get_int_id(doc_id), 0))
    return train_set


class Antique(Dataset):
    """Antique dataset class."""
    def _read_dataset(self):
        """Read all dataset files.

        Returns:
            tuple[dict[int, str], dict[int, str], dict[int, set[int]], set[int],
                  dict[int, tuple[int, int]], dict[int, tuple[int, int]]] -- A tuple containing
                * a mapping of query IDs to queries
                * a mapping of document IDs to documents
                * a mapping of train query IDs to tuples of (document ID, label)
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
        train_set = read_train_set(train_qrels_file)

        test_queries_file = os.path.join(self.args.ANTIQUE_DIR, 'antique-test-queries.txt')
        test_queries = read_queries(test_queries_file)

        queries = train_queries.copy()
        queries.update(test_queries)

        print('reading {}...'.format(self.args.SPLIT_FILE))
        with open(self.args.SPLIT_FILE, 'rb') as fp:
            train_q_ids, dev_set, test_set = pickle.load(fp)
        train_set = {q_id: train_set[q_id] for q_id in train_q_ids}

        return queries, docs, train_set, dev_set, test_set

    @staticmethod
    def add_subparser(subparsers, name):
        """Add a dataset-specific subparser with all required arguments."""
        sp = subparsers.add_parser(name)
        sp.add_argument('ANTIQUE_DIR', help='Folder with all Antique files')
        sp.add_argument('SPLIT_FILE', help='File with train/dev/test split')
