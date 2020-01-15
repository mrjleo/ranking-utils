import os
import csv
import pickle
from collections import defaultdict

from tqdm import tqdm

from qa_utils.preprocessing.misc import count_lines
from qa_utils.preprocessing.dataset import Dataset


def read_collection(file_path):
    print('processing {}...'.format(file_path), flush=True)
    items = {}
    total = count_lines(file_path)
    with open(file_path, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for item_id, item in tqdm(reader, total=total):
            items[int(item_id)] = item
    return items


def read_qrels(file_path):
    print('processing {}...'.format(file_path), flush=True)
    qrels = defaultdict(set)
    total = count_lines(file_path)
    with open(file_path, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for q_id, _, doc_id, _ in tqdm(reader, total=total):
            qrels[int(q_id)].add(int(doc_id))
    return qrels


def read_dev_set(dev_set_file, qrels_file):
    print('processing {}...'.format(qrels_file), flush=True)
    qrels = defaultdict(set)
    total = count_lines(qrels_file)
    with open(qrels_file, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for q_id, _, doc_id, _ in tqdm(reader, total=total):
            qrels[int(q_id)].add(int(doc_id))

    print('processing {}...'.format(dev_set_file), flush=True)
    dev_set = defaultdict(list)
    total = count_lines(dev_set_file)
    with open(dev_set_file, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for q_id, doc_id, _, _ in tqdm(reader, total=total):
            label = 1 if int(doc_id) in qrels[int(q_id)] else 0
            dev_set[int(q_id)].append((int(doc_id), label))
    return dev_set


class MSMARCO(Dataset):
    """MS MARCO passage ranking dataset class."""
    def _read_dataset(self):
        """Read all dataset files.

        Returns:
            tuple[dict[int, str], dict[int, str], dict[int, set[int]], set[int],
                  dict[int, tuple[int, int]], dict[int, tuple[int, int]]] -- A tuple containing
                * a mapping of query IDs to queries
                * a mapping of document IDs to documents
                * a mapping of query IDs to relevant document IDs
                * a set of query IDs in the trainset
                * a mapping of dev query IDs to tuples of (document ID, label)
                * a mapping of test query IDs to tuples of (document ID, label)
        """
        docs_file = os.path.join(self.args.MSM_DIR, 'collection.tsv')
        train_queries_file = os.path.join(self.args.MSM_DIR, 'queries.train.tsv')
        train_qrels_file = os.path.join(self.args.MSM_DIR, 'qrels.train.tsv')
        docs = read_collection(docs_file)
        train_queries = read_collection(train_queries_file)
        qrels = read_qrels(train_qrels_file)

        orig_dev_file = os.path.join(self.args.MSM_DIR, 'top1000.dev.tsv')
        orig_dev_qrels_file = os.path.join(self.args.MSM_DIR, 'qrels.dev.tsv')
        orig_dev_set = read_dev_set(orig_dev_file, orig_dev_qrels_file)
        orig_dev_queries_file = os.path.join(self.args.MSM_DIR, 'queries.dev.tsv')
        orig_dev_queries = read_collection(orig_dev_queries_file)

        queries = train_queries.copy()
        queries.update(orig_dev_queries)
        train_q_ids = train_queries.keys()

        print('reading {}...'.format(self.args.MSM_SPLIT))
        with open(self.args.MSM_SPLIT, 'rb') as fp:
            dev_q_ids = pickle.load(fp)

        dev_set, test_set = {}, {}
        for q_id, d in orig_dev_set.items():
            if q_id in dev_q_ids:
                dev_set[q_id] = d
            else:
                test_set[q_id] = d

        return queries, docs, qrels, train_q_ids, dev_set, test_set

    @staticmethod
    def add_subparser(subparsers, name):
        """Add a dataset-specific subparser with all required arguments."""
        sp = subparsers.add_parser(name)
        sp.add_argument('MSM_DIR', help='Folder with all MS MARCO ranking files')
        sp.add_argument('MSM_SPLIT', help='MS Marco split')
