import csv
import gzip
import argparse
from pathlib import Path
from collections import defaultdict

from qa_utils.datasets.dataset import Dataset


class InsuranceQA(Dataset):
    """InsuranceQA dataset class.

    Args:
        args (argparse.Namespace): Namespace that contains the arguments defined below
    """
    def __init__(self, args: argparse.Namespace):
        base_dir = Path(args.INSRQA_V2_DIR)

        vocab_file = base_dir / 'vocabulary'
        print(f'reading {vocab_file}...')
        vocab = {}
        with open(vocab_file, encoding='utf-8') as fp:
            for idx, word in csv.reader(fp, delimiter='\t', quotechar=None):
                vocab[idx] = word

        def _decode(idx_list):
            return ' '.join(map(vocab.get, idx_list))

        l2a_file = base_dir / 'InsuranceQA.label2answer.token.encoded.gz'
        print(f'reading {l2a_file}...')
        docs = {}
        with gzip.open(l2a_file) as fp:
            for line in fp:
                doc_id, doc_idxs = line.decode('utf-8').split('\t')
                docs[doc_id] = _decode(doc_idxs.split())

        # read all qrels and top documents
        files = [
            base_dir / f'InsuranceQA.question.anslabel.token.{args.examples_per_query}.pool.solr.train.encoded.gz',
            base_dir / f'InsuranceQA.question.anslabel.token.{args.examples_per_query}.pool.solr.valid.encoded.gz',
            base_dir / f'InsuranceQA.question.anslabel.token.{args.examples_per_query}.pool.solr.test.encoded.gz'
        ]
        sets = [set(), set(), set()]
        prefixes = ['train', 'val', 'test']

        queries, qrels, pools = {}, defaultdict(dict), defaultdict(set)
        for f, ids, prefix in zip(files, sets, prefixes):
            print(f'reading {f}...')
            with gzip.open(f) as fp:
                for i, line in enumerate(fp):
                    # we need to make the query IDs unique
                    q_id = f'{prefix}_{i}'
                    _, q_idxs, gt, pool = line.decode('utf-8').split('\t')
                    queries[q_id] = _decode(q_idxs.split())

                    ids.add(q_id)
                    for doc_id in gt.split():
                        qrels[q_id][doc_id] = 1
                    for doc_id in pool.split():
                        pools[q_id].add(doc_id)

        train_ids, val_ids, test_ids = sets
        super().__init__(queries, docs, qrels, pools, train_ids, val_ids, test_ids)


    @staticmethod
    def add_subparser(subparsers: argparse._SubParsersAction, name: str):
        """Add a dataset-specific subparser with all required arguments.

        Args:
            subparsers (argparse._SubParsersAction): Subparsers to add a parser to
            name (str): Parser name
        """
        sp = subparsers.add_parser(name)
        sp.add_argument('INSRQA_V2_DIR', help='InsuranceQA V2 dataset directory')
        sp.add_argument('--examples_per_query', type=int, choices=[100, 500, 1000, 1500],
                        default=500, help='How many examples per query in the dev- and testset')
