import os
import csv
import gzip
import pickle
import argparse
from collections import defaultdict

from tqdm import tqdm

from qa_utils.preprocessing.misc import count_lines
from qa_utils.preprocessing.dataset import Dataset


def decode(idx_list, vocab):
    return ' '.join([vocab[idx].lower() for idx in idx_list])


class InsuranceQA(Dataset):
    """InsuranceQA dataset class."""    
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
        vocab_file = os.path.join(self.args.INSRQA_V2_DIR, 'vocabulary')
        l2a_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.label2answer.token.encoded.gz')
        # use the smallest file here as we do the sampling by ourselves
        train_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded.gz')
        # use dev- and test-set with 1000 examples per query
        dev_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.question.anslabel.token.1000.pool.solr.valid.encoded.gz')
        test_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.question.anslabel.token.1000.pool.solr.test.encoded.gz')

        print('processing {}...'.format(vocab_file), flush=True)
        vocab = {}
        total = count_lines(vocab_file)
        with open(vocab_file, encoding='utf-8') as fp:
            for idx, word in tqdm(csv.reader(fp, delimiter='\t', quotechar=None), total=total):
                vocab[idx] = word

        print('processing {}...'.format(l2a_file), flush=True)
        docs = {}
        total = count_lines(l2a_file)
        with gzip.open(l2a_file) as fp:
            for line in tqdm(fp, total=total):
                doc_id, doc_idxs = line.decode('utf-8').split('\t')
                docs[int(doc_id)] = decode(doc_idxs.split(), vocab)

        print('processing {}...'.format(train_file), flush=True)
        queries = {}
        qrels = defaultdict(set)
        total = count_lines(train_file)
        with gzip.open(train_file) as fp:
            for q_id, line in enumerate(tqdm(fp, total=total)):
                _, q_idxs, gt, _ = line.decode('utf-8').split('\t')
                queries[q_id] = decode(q_idxs.split(), vocab)
                qrels[q_id] = set(map(int, gt.split()))
        train_q_ids = qrels.keys()

        print('processing {}...'.format(dev_file), flush=True)
        dev_set = defaultdict(list)
        total = count_lines(dev_file)
        with gzip.open(dev_file) as fp:
            for q_id, line in enumerate(tqdm(fp, total=total)):
                _, q_idxs, gt, pool = line.decode('utf-8').split('\t')
                queries[q_id] = decode(q_idxs.split(), vocab)

                pos_doc_ids = set(map(int, gt.split()))
                # make sure no positive IDs are in the pool
                neg_doc_ids = set(map(int, pool.split())) - pos_doc_ids
                assert len(pos_doc_ids & neg_doc_ids) == 0

                for pos_doc_id in pos_doc_ids:
                    dev_set[q_id].append((pos_doc_id, 1))
                for neg_doc_id in neg_doc_ids:
                    dev_set[q_id].append((neg_doc_id, 0)) 

        print('processing {}...'.format(test_file), flush=True)
        test_set = defaultdict(list)
        total = count_lines(test_file)
        with gzip.open(test_file) as fp:
            for q_id, line in enumerate(tqdm(fp, total=total)):
                _, q_idxs, gt, pool = line.decode('utf-8').split('\t')
                queries[q_id] = decode(q_idxs.split(), vocab)

                pos_doc_ids = set(map(int, gt.split()))
                # make sure no positive IDs are in the pool
                neg_doc_ids = set(map(int, pool.split())) - pos_doc_ids
                assert len(pos_doc_ids & neg_doc_ids) == 0

                for pos_doc_id in pos_doc_ids:
                    test_set[q_id].append((pos_doc_id, 1))
                for neg_doc_id in neg_doc_ids:
                    test_set[q_id].append((neg_doc_id, 0))

        return queries, docs, qrels, train_q_ids, dev_set, test_set 


    @staticmethod
    def add_subparser(subparsers, name):
        """Add a dataset-specific subparser with all required arguments."""
        sp = subparsers.add_parser(name)
        sp.add_argument('INSRQA_V2_DIR', help='Folder with insuranceQA v2 files')
