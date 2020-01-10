import os
import csv
import gzip
from collections import defaultdict

from tqdm import tqdm

from qa_utils.preprocessing.misc import count_lines
from qa_utils.preprocessing.dataset import Dataset


def decode(idx_list, vocab):
    return ' '.join(map(vocab.get, idx_list))


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
        num = self.args.examples_per_query
        vocab_file = os.path.join(self.args.INSRQA_V2_DIR, 'vocabulary')
        l2a_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.label2answer.token.encoded.gz')
        # use the smallest file here as we do the sampling by ourselves
        train_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded.gz')
        dev_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.question.anslabel.token.{}.pool.solr.valid.encoded.gz'.format(num))
        test_file = os.path.join(self.args.INSRQA_V2_DIR,
            'InsuranceQA.question.anslabel.token.{}.pool.solr.test.encoded.gz'.format(num))

        print('processing {}...'.format(vocab_file))
        vocab = {}
        total = count_lines(vocab_file)
        with open(vocab_file, encoding='utf-8') as fp:
            for idx, word in tqdm(csv.reader(fp, delimiter='\t', quotechar=None), total=total):
                vocab[idx] = word

        print('processing {}...'.format(l2a_file))
        docs = {}
        total = count_lines(l2a_file)
        with gzip.open(l2a_file) as fp:
            for line in tqdm(fp, total=total):
                doc_id, doc_idxs = line.decode('utf-8').split('\t')
                docs[int(doc_id)] = decode(doc_idxs.split(), vocab)

        print('processing {}...'.format(train_file))
        queries = {}
        qrels = defaultdict(set)
        total = count_lines(train_file)
        with gzip.open(train_file) as fp:
            for i, line in enumerate(tqdm(fp, total=total)):
                q_id = i

                _, q_idxs, gt, _ = line.decode('utf-8').split('\t')
                queries[q_id] = decode(q_idxs.split(), vocab)
                qrels[q_id] = set(map(int, gt.split()))
        train_q_ids = qrels.keys()

        print('processing {}...'.format(dev_file))
        dev_set = defaultdict(list)
        total = count_lines(dev_file)
        with gzip.open(dev_file) as fp:
            for i, line in enumerate(tqdm(fp, total=total)):
                # we need to make the q_id unique
                # since the dataset has less than 20k queries, this works
                q_id = 100000 + i

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

        print('processing {}...'.format(test_file))
        test_set = defaultdict(list)
        total = count_lines(test_file)
        with gzip.open(test_file) as fp:
            for i, line in enumerate(tqdm(fp, total=total)):
                # we need to make the q_id unique
                # since the dataset has less than 20k queries, this works
                q_id = 200000 + i

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
        sp.add_argument('--examples_per_query', type=int, choices=[100, 500, 1000, 1500],
                        default=500, help='How many examples per query in the dev- and testset')
