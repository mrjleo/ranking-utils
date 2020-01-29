import os
import csv
import pickle
from collections import defaultdict

from tqdm import tqdm

from qa_utils.preprocessing.misc import count_lines
from qa_utils.preprocessing.dataset import Dataset


def get_doc_list(q_id, top, qrels):
    result = []
    # positives
    for doc_id in qrels[q_id]:
        result.append((doc_id, 1))
    # remaining negatives
    for doc_id in top[q_id]:
        if doc_id not in qrels[q_id]:
            result.append((doc_id, 0))
    return result


class FiQA(Dataset):
    """FiQA dataset class."""
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
        question_file = os.path.join(self.args.FIQA_DIR, 'FiQA_train_question_final.tsv')
        doc_file = os.path.join(self.args.FIQA_DIR, 'FiQA_train_doc_final.tsv')
        question_doc_file = os.path.join(self.args.FIQA_DIR, 'FiQA_train_question_doc_final.tsv')
        split_file = self.args.SPLIT_FILE

        print('reading {}...'.format(question_file))
        queries = {}
        total = count_lines(question_file) - 1
        with open(question_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for _, q_id, question, _ in tqdm(csv.reader(fp, delimiter='\t'), total=total):
                queries[int(q_id)] = question

        print('reading {}...'.format(doc_file))
        docs = {}
        total = count_lines(doc_file) - 1
        with open(doc_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for _, doc_id, doc, _ in tqdm(csv.reader(fp, delimiter='\t'), total=total):
                docs[int(doc_id)] = doc

        print('reading {}...'.format(question_doc_file))
        qrels = defaultdict(set)
        total = count_lines(question_doc_file) - 1
        with open(question_doc_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for _, q_id, doc_id in tqdm(csv.reader(fp, delimiter='\t'), total=total):
                qrels[int(q_id)].add(int(doc_id))

        print('reading {}...'.format(split_file))
        with open(split_file, 'rb') as fp:
            top, dev_q_ids, test_q_ids = pickle.load(fp)
        assert len(queries) == len(top)

        train_set, dev_set, test_set = {}, {}, {}
        for q_id in queries:
            if q_id in dev_q_ids:
                dev_set[q_id] = get_doc_list(q_id, top, qrels)
            elif q_id in test_q_ids:
                test_set[q_id] = get_doc_list(q_id, top, qrels)
            else:
                train_set[q_id] = get_doc_list(q_id, top, qrels)

        return queries, docs, train_set, dev_set, test_set

    @staticmethod
    def add_subparser(subparsers, name):
        """Add a dataset-specific subparser with all required arguments."""
        sp = subparsers.add_parser(name)
        sp.add_argument('FIQA_DIR', help='Folder with all FiQA files')
        sp.add_argument('SPLIT_FILE', help='File with train/dev/test split')
