import csv
from pathlib import Path
from typing import Dict

__version__ = "0.1.0"


def write_trec_eval_file(
    out_file: Path, predictions: Dict[str, Dict[str, float]], name: str
) -> None:
    """Write the results in a file accepted by the TREC evaluation tool.

    Args:
        out_file (Path): File to create.
        predictions (Dict[str, Dict[str, float]]): Query IDs mapped to document IDs mapped to scores.
        name (str): Method name.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for q_id in predictions:
            ranking = sorted(
                predictions[q_id].keys(), key=predictions[q_id].get, reverse=True
            )
            for rank, doc_id in enumerate(ranking, 1):
                score = predictions[q_id][doc_id]
                writer.writerow([q_id, "Q0", doc_id, rank, score, name])
