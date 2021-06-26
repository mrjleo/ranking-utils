#! /usr/bin/env python3


import argparse
from pathlib import Path

from ranking_utils.util import read_predictions, write_trec_eval_file


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('PREDICTION_FILES', nargs='+', help='Prediction files (.pkl)')
    ap.add_argument('--out_file', default='out.tsv', help='Output file to use with TREC-eval')
    ap.add_argument('--name', default='(none)', help='Method name')
    args = ap.parse_args()

    predictions = read_predictions(map(Path, args.PREDICTION_FILES))
    write_trec_eval_file(Path(args.out_file), predictions, args.name)


if __name__ == '__main__':
    main()
