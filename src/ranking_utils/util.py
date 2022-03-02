import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import h5py
import torch
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter

from ranking_utils.lightning.ranker import Ranker
from ranking_utils.lightning.data import RankingDataModule


# class CustomWriter(BasePredictionWriter):
#     def __init__(self, temp_dir: Path):
#         super().__init__("epoch")
#         self.temp_dir = temp_dir

#     def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
#         out_file = self.temp_dir / f"out_{trainer.global_rank}"
#         torch.save(predictions, out_file)


def predict_and_save(
    ranker: Ranker,
    data_module: RankingDataModule,
    out_file: Path,
    name: str,
    accelerator="auto",
    devices="auto",
    strategy="dp",
):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(accelerator=accelerator, devices=devices, strategy=strategy)
    result = defaultdict(dict)
    for predictions in trainer.predict(model=ranker, datamodule=data_module):
        for index, score in zip(predictions["indices"], predictions["scores"]):
            q_id, doc_id = data_module.data_provider.get_prediction_ids(index)
            result[q_id][doc_id] = score.numpy()
    write_trec_eval_file(out_file, result, name)


def write_trec_eval_file(
    out_file: Path, predictions: Dict[str, Dict[str, float]], name: str
):
    """Write the results in a file accepted by the TREC evaluation tool.

    Args:
        out_file (Path): File to create
        predictions (Dict[str, Dict[str, float]]): Query IDs mapped to document IDs mapped to scores
        name (str): Method name
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


# def create_temp_testsets(
#     data_file: Path, runfiles: Iterable[Path]
# ) -> List[Tuple[int, str]]:
#     """Create re-ranking testsets in a temporary files.

#     Args:
#         data_file (Path): Pre-processed data file containing queries and documents
#         runfiles (Iterable[Path]): Runfiles to create testsets for (TREC format)

#     Returns:
#         List[Tuple[int, str]]: Descriptors and paths of the temporary files
#     """
#     # recover the internal integer query and doc IDs
#     int_q_ids = {}
#     int_doc_ids = {}
#     with h5py.File(data_file, "r") as fp:
#         for int_id, orig_id in enumerate(
#             tqdm(fp["orig_q_ids"].asstr(), total=len(fp["orig_q_ids"]))
#         ):
#             int_q_ids[orig_id] = int_id
#         for int_id, orig_id in enumerate(
#             tqdm(fp["orig_doc_ids"].asstr(), total=len(fp["orig_doc_ids"]))
#         ):
#             int_doc_ids[orig_id] = int_id

#     result = []
#     for runfile in runfiles:
#         qd_pairs = []
#         with open(runfile) as fp:
#             for line in fp:
#                 q_id, _, doc_id, _, _, _ = line.split()
#                 qd_pairs.append((q_id, doc_id))
#         fd, f = tempfile.mkstemp()
#         with h5py.File(f, "w") as fp:
#             num_items = len(qd_pairs)
#             ds = {
#                 "q_ids": fp.create_dataset("q_ids", (num_items,), dtype="int32"),
#                 "doc_ids": fp.create_dataset("doc_ids", (num_items,), dtype="int32"),
#                 "labels": fp.create_dataset("labels", (num_items,), dtype="int32"),
#             }
#             for i, (q_id, doc_id) in enumerate(tqdm(qd_pairs, desc="Saving testset")):
#                 ds["q_ids"][i] = int_q_ids[q_id]
#                 ds["doc_ids"][i] = int_doc_ids[doc_id]
#         result.append((fd, f))
#     return result
