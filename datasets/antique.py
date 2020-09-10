import csv
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

from qa_utils.datasets.dataset import Dataset


class ANTIQUE(Dataset):
    """ANTIQUE dataset class.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments defined below
        num_negatives (int): Number of negative examples
    """
    def __init__(self, args: argparse.Namespace, num_negatives: int):
        base_dir = Path(args.ANTIQUE_DIR)
        split_file = Path(args.SPLIT_FILE)

        # read all queries
        queries = {}
        for f_name in ['antique-train-queries.txt', 'antique-test-queries.txt']:
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                queries.update({q_id: query for q_id, query in csv.reader(fp, delimiter='\t')})

        # read documents
        doc_file = base_dir / 'antique-collection.txt'
        print(f'reading {doc_file}...')
        with open(doc_file, encoding='utf-8') as fp:
            docs = {doc_id: doc for doc_id, doc in csv.reader(fp, delimiter='\t', quotechar=None)}

        # read all qrels
        qrels = defaultdict(set)
        q_ids = defaultdict(set)
        for f_name in ['antique-train.qrel', 'antique-test.qrel']:
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                for line in fp:
                    q_id, _, doc_id, rel = line.split()

                    # authors recommend treating rel > 2 as positive
                    if int(rel) > 2:
                        qrels[q_id].add(doc_id)

                    q_ids[f_name].add(q_id)

        print(f'reading {split_file}...')
        with open(split_file, 'rb') as fp:
            top, val_ids = pickle.load(fp)

        train_ids = q_ids['antique-train.qrel'] - val_ids
        test_ids = q_ids['antique-test.qrel']
        super().__init__(queries, docs, qrels, top, train_ids, val_ids, test_ids, num_negatives)

    @staticmethod
    def add_subparser(subparsers: argparse._SubParsersAction, name: str):
        """Add a dataset-specific subparser with all required arguments.

        Args:
            subparsers (argparse._SubParsersAction): Subparsers to add a parser to
            name (str): Parser name
        """
        sp = subparsers.add_parser(name)
        sp.add_argument('ANTIQUE_DIR', help='Folder with all Antique files')
        sp.add_argument('SPLIT_FILE', help='File with train/dev/test split')
