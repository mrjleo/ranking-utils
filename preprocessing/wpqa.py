import os
import csv
import json
import pickle
from collections import defaultdict

from tqdm import tqdm

from qa_utils.preprocessing.misc import count_lines
from qa_utils.preprocessing.dataset import Dataset


class WikiPassageQA(Dataset):
    """WikiPassageQA dataset class."""
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
        doc_file = os.path.join(self.args.WPQA_DIR, 'document_passages.json')
        train_file = os.path.join(self.args.WPQA_DIR, 'train.tsv')
        dev_file = os.path.join(self.args.WPQA_DIR, 'dev.tsv')
        test_file = os.path.join(self.args.WPQA_DIR, 'test.tsv')

        print('processing {}...'.format(doc_file), flush=True)
        with open(doc_file, encoding='utf-8') as fp:
            docs_json = json.load(fp)
        docs = {}
        for art_id in tqdm(docs_json):
            for p_id, passage in docs_json[art_id].items():
                docs[(int(art_id), int(p_id))] = passage

        print('processing {}...'.format(train_file), flush=True)
        queries = {}
        qrels = defaultdict(set)
        total = count_lines(train_file) - 1
        with open(train_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for q_id, q, art_id, _, p_ids in tqdm(csv.reader(fp, delimiter='\t'), total=total):
                queries[int(q_id)] = q
                for p_id in map(int, p_ids.split(',')):
                    qrels[int(q_id)].add((int(art_id), p_id))

        print('processing {}...'.format(dev_file), flush=True)
        total = count_lines(dev_file) - 1
        with open(dev_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for q_id, q, _, _, _ in tqdm(csv.reader(fp, delimiter='\t'), total=total):
                queries[int(q_id)] = q

        print('processing {}...'.format(test_file), flush=True)
        total = count_lines(test_file) - 1
        with open(test_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for q_id, q, _, _, _ in tqdm(csv.reader(fp, delimiter='\t'), total=total):
                queries[int(q_id)] = q

        train_q_ids = qrels.keys()
        with open(self.args.SPLIT_FILE, 'rb') as fp:
            dev_set, test_set = pickle.load(fp)

        return queries, docs, qrels, train_q_ids, dev_set, test_set

    @staticmethod
    def add_subparser(subparsers, name):
        """Add a dataset-specific subparser with all required arguments."""
        sp = subparsers.add_parser(name)
        sp.add_argument('WPQA_DIR', help='Folder with all WikiPassageQA files')
        sp.add_argument('SPLIT_FILE', help='File with train/dev/test split')
