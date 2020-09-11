import csv
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

from qa_utils.datasets.dataset import Dataset


class FiQA(Dataset):
    """FiQA dataset class.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments defined below
        num_negatives (int): Number of negative examples
    """
    def __init__(self, args: argparse.Namespace, num_negatives: int):
        base_dir = Path(args.FIQA_DIR)
        split_file = Path(args.SPLIT_FILE)

        question_file = base_dir / 'FiQA_train_question_final.tsv'
        print(f'reading {question_file}...')
        queries = {}
        with open(question_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for _, q_id, question, _ in csv.reader(fp, delimiter='\t'):
                queries[q_id] = question

        doc_file = base_dir / 'FiQA_train_doc_final.tsv'
        print(f'reading {doc_file}...')
        docs = {}
        with open(doc_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for _, doc_id, doc, _ in csv.reader(fp, delimiter='\t'):
                docs[doc_id] = doc

        question_doc_file = base_dir / 'FiQA_train_question_doc_final.tsv'
        print(f'reading {question_doc_file}...')
        qrels = defaultdict(set)
        with open(question_doc_file, encoding='utf-8') as fp:
            # skip header
            next(fp)
            for _, q_id, doc_id in csv.reader(fp, delimiter='\t'):
                qrels[q_id].add(doc_id)

        print(f'reading {split_file}...')
        with open(split_file, 'rb') as fp:
            top, val_ids, test_ids = pickle.load(fp)
        assert len(queries) == len(top)

        train_ids = set(queries.keys()) - val_ids - test_ids
        super().__init__(queries, docs, qrels, top, train_ids, val_ids, test_ids, num_negatives)

    @staticmethod
    def add_subparser(subparsers: argparse._SubParsersAction, name: str):
        """Add a dataset-specific subparser with all required arguments.

        Args:
            subparsers (argparse._SubParsersAction): Subparsers to add a parser to
            name (str): Parser name
        """
        sp = subparsers.add_parser(name)
        sp.add_argument('FIQA_DIR', help='Folder with all FiQA files')
        sp.add_argument('SPLIT_FILE', help='File with train/dev/test split')
