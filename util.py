import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Tuple

import torch


def read_output_files(files: Iterable[Path]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """Read output files created during testing.

    Args:
        files (Iterable[Path]): Output files, typically one per GPU

    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]: Query IDs mapped to document IDs mapped to scores/labels
    """
    predictions, labels = defaultdict(dict), defaultdict(dict)
    for f in files:
        for d in torch.load(f):
            predictions[d['q_id']][d['doc_id']] = d['prediction'][0]
            labels[d['q_id']][d['doc_id']] = d['label']
    return predictions, labels


def write_trec_eval_file(out_file: Path, predictions: Dict[str, Dict[str, float]], name: str):
    """Write the results in a file accepted by the TREC evaluation tool.

    Args:
        out_file (Path): File to create
        predictions (Dict[str, Dict[str, float]]): Query IDs mapped to document IDs mapped to scores
        name (str): Method name
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8', newline='\n') as fp:
        writer = csv.writer(fp, delimiter='\t')
        for q_id in predictions:
            ranking = sorted(predictions[q_id].keys(), key=predictions[q_id].get, reverse=True)
            for rank, doc_id in enumerate(ranking, 1):
                score = predictions[q_id][doc_id]
                writer.writerow([q_id, 'Q0', doc_id, rank, score, name])
