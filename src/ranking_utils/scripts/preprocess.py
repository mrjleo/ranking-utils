#! /usr/bin/env python3


import argparse
from pathlib import Path

from pytorch_lightning import seed_everything

from ranking_utils.datasets.antique import ANTIQUE
from ranking_utils.datasets.fiqa import FiQA
from ranking_utils.datasets.insuranceqa import InsuranceQA
from ranking_utils.datasets.trecdl import TRECDL2019Passage, TRECDL2019Document
from ranking_utils.datasets.trec import TREC


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("SAVE", help="Where to save the results")
    ap.add_argument(
        "--num_neg_point",
        type=int,
        default=1,
        help="Number of negatives per positive (pointwise training)",
    )
    ap.add_argument(
        "--num_neg_pair",
        type=int,
        default=16,
        help="Number of negatives per positive (pairwise training)",
    )
    ap.add_argument(
        "--query_limit_pair",
        type=int,
        default=64,
        help="Maximum number of training examples per query (pairwise training)",
    )
    ap.add_argument("--random_seed", type=int, default=123, help="Random seed")

    subparsers = ap.add_subparsers(help="Choose a dataset", dest="dataset")
    subparsers.required = True
    DATASETS = [ANTIQUE, FiQA, InsuranceQA, TRECDL2019Passage, TRECDL2019Document, TREC]
    for c in DATASETS:
        c.add_subparser(subparsers, c.__name__.lower())
    args = ap.parse_args()

    if args.random_seed:
        seed_everything(args.random_seed)

    ds = None
    for c in DATASETS:
        if args.dataset == c.__name__.lower():
            ds = c(args)
            break

    save_path = Path(args.SAVE)
    ds.save(save_path, args.num_neg_point, args.num_neg_pair, args.query_limit_pair)


if __name__ == "__main__":
    main()
