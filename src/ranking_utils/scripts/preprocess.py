#! /usr/bin/env python3


import argparse
import logging
from pathlib import Path

from pytorch_lightning import seed_everything
from ranking_utils.datasets.antique import ANTIQUE
from ranking_utils.datasets.fiqa import FiQA
from ranking_utils.datasets.insuranceqa import InsuranceQA
from ranking_utils.datasets.msmarco import MSMARCODocument, MSMARCOPassage
from ranking_utils.datasets.trec import TREC

DATASETS = {
    c.__name__.lower(): c
    for c in [ANTIQUE, FiQA, InsuranceQA, MSMARCOPassage, MSMARCODocument, TREC]
}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("SAVE", type=Path, help="Where to save the results.")
    ap.add_argument(
        "DATASET", choices=DATASETS.keys(), help="Which dataset to process."
    )
    ap.add_argument(
        "ROOT_DIR", type=Path, help="Directory that contains all dataset files."
    )
    ap.add_argument(
        "--num_neg_point",
        type=int,
        default=1,
        help="Number of negatives per positive (pointwise training).",
    )
    ap.add_argument(
        "--num_neg_pair",
        type=int,
        default=1,
        help="Number of negatives per positive (pairwise training).",
    )
    ap.add_argument(
        "--query_limit_pair",
        type=int,
        default=100,
        help="Maximum number of training examples per query (pairwise training).",
    )
    ap.add_argument("--random_seed", type=int, default=123, help="Random seed.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.random_seed:
        seed_everything(args.random_seed)

    DATASETS[args.DATASET](args.ROOT_DIR).save(
        args.SAVE, args.num_neg_point, args.num_neg_pair, args.query_limit_pair
    )


if __name__ == "__main__":
    main()
