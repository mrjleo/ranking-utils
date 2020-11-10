import csv
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from qa_utils.datasets.dataset import Dataset


class MSMARCO(Dataset):
    """MS MARCO passage ranking dataset class. Uses the testset of the 2019 TREC DL track.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments defined below
        num_negatives (int): Number of negatives per positive
        query_limit (int): Maximum number of training examples per query
    """
    def __init__(self, args: argparse.Namespace, num_negatives: int, query_limit: int):
        base_dir = Path(args.MSMARCO_DIR)

        # read queries
        queries = {}
        for f_name, num_lines in [('queries.train.tsv', 808731),
                                  ('queries.dev.tsv', 101093),
                                  ('msmarco-test2019-queries.tsv', 200)]:
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, query in tqdm(reader, total=num_lines):
                    queries[q_id] = query

        # read documents
        docs = {}
        f = base_dir / 'collection.tsv'
        print(f'reading {f}...')
        with open(f, encoding='utf-8') as fp:
            reader = csv.reader(fp, delimiter='\t')
            for doc_id, doc in tqdm(reader, total=8841823):
                docs[doc_id] = doc

        # read qrels
        qrels = defaultdict(dict)
        q_ids = defaultdict(set)
        for f_name, num_lines in [('qrels.train.tsv', 532761), ('qrels.dev.tsv', 59273)]:
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, _, doc_id, rel in tqdm(reader, total=num_lines):
                    qrels[q_id][doc_id] = int(rel)
                    q_ids[f_name].add(q_id)

        # TREC qrels have a different format
        f = base_dir / '2019qrels-pass.txt'
        print(f'reading {f}...')
        with open(f, encoding='utf-8') as fp:
            for q_id, _, doc_id, rel in csv.reader(fp, delimiter=' '):
                # 1 is considered irrelevant
                qrels[q_id][doc_id] = int(rel) - 1
                q_ids['2019qrels-pass.txt'].add(q_id)

        # read top documents
        pools = defaultdict(set)
        for f_name, num_lines in [('top1000.train.txt', 478016942),
                                  ('top1000.dev.tsv', 6668967),
                                  ('msmarco-passagetest2019-top1000.tsv', 189877)]:
            f = base_dir / f_name
            print(f'reading {f}...')
            with open(f, encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter='\t')
                for q_id, doc_id, _, _ in tqdm(reader, total=num_lines):
                    pools[q_id].add(doc_id)

        # some IDs have no pool or no query -- remove them
        all_ids = set(pools.keys()) & set(queries.keys())
        train_ids = q_ids['qrels.train.tsv'] & all_ids
        val_ids = q_ids['qrels.dev.tsv'] & all_ids
        test_ids = q_ids['2019qrels-pass.txt'] & all_ids
        super().__init__(queries, docs, qrels, pools, train_ids, val_ids, test_ids, num_negatives, query_limit)

    @staticmethod
    def add_subparser(subparsers: argparse._SubParsersAction, name: str):
        """Add a dataset-specific subparser with all required arguments.

        Args:
            subparsers (argparse._SubParsersAction): Subparsers to add a parser to
            name (str): Parser name
        """
        sp = subparsers.add_parser(name)
        sp.add_argument('MSMARCO_DIR', help='MS MARCO passage ranking dataset directory')
