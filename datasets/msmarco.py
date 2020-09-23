import csv
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from qa_utils.datasets.dataset import Dataset


class MSMARCO(Dataset):
    """MS MARCO passage ranking dataset class.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments defined below
        num_negatives (int): Number of negative examples
    """
    def __init__(self, args: argparse.Namespace, num_negatives: int):
        base_dir = Path(args.MSMARCO_DIR)
        split_file = Path(args.SPLIT_FILE)

        # read all queries
        queries = {}
        for f_name, num_lines in zip(['queries.train.tsv', 'queries.dev.tsv'], [808731, 101093]):
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, query in tqdm(reader, total=num_lines):
                    queries[q_id] = query

        # read documents
        docs = {}
        docs_file = base_dir / 'collection.tsv'
        print(f'reading {docs_file}...')
        with open(docs_file, encoding='utf-8') as fp:
            reader = csv.reader(fp, delimiter='\t')
            for doc_id, doc in tqdm(reader, total=8841823):
                docs[doc_id] = doc

        # read all qrels
        qrels = defaultdict(set)
        q_ids = defaultdict(set)
        for f_name, num_lines in zip(['qrels.train.tsv', 'qrels.dev.tsv'], [532761, 59273]):
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, _, doc_id, _ in tqdm(reader, total=num_lines):
                    qrels[q_id].add(doc_id)
                    q_ids[f_name].add(q_id)

        # read all top documents
        pools = defaultdict(set)
        for f_name, num_lines in zip(['top1000.train.txt', 'top1000.dev.tsv'], [478016942, 6668967]):
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, doc_id, _, _ in tqdm(reader, total=num_lines):
                    pools[q_id].add(doc_id)

        print(f'reading {split_file}...')
        with open(split_file, 'rb') as fp:
            val_ids = pickle.load(fp)

        train_ids = q_ids['qrels.train.tsv']
        test_ids = q_ids['qrels.dev.tsv'] - val_ids
        super().__init__(queries, docs, qrels, pools, train_ids, val_ids, test_ids, num_negatives)

    @staticmethod
    def add_subparser(subparsers: argparse._SubParsersAction, name: str):
        """Add a dataset-specific subparser with all required arguments.

        Args:
            subparsers (argparse._SubParsersAction): Subparsers to add a parser to
            name (str): Parser name
        """
        sp = subparsers.add_parser(name)
        sp.add_argument('MSMARCO_DIR', help='MS MARCO passage ranking dataset directory')
        sp.add_argument('SPLIT_FILE', help='MS MARCO split')
