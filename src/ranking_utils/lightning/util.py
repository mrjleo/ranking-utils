import pickle
from pathlib import Path
from typing import Dict, Iterable
from collections import defaultdict

from pytorch_lightning import Trainer

from ranking_utils.lightning.datasets import ValTestDatasetBase


def predict_and_save(trainer: Trainer, test_ds: ValTestDatasetBase):
    """Predict and save predictions in a file. The file is created in the `log_dir` of the trainer.
    Original query and document IDs are recovered and written in the files.
    The file name is unique w.r.t. `trainer.local_rank`.

    Args:
        trainer (Trainer): Trainer object with associated model
        test_ds (ValTestDatasetBase): Test dataset used to recover original IDs
    """
    out_dict = defaultdict(list)
    for item in trainer.predict():
        out_dict['q_ids'].extend(map(test_ds.get_original_query_id, item['q_ids'].tolist()))
        out_dict['doc_ids'].extend(map(test_ds.get_original_document_id, item['doc_ids'].tolist()))
        out_dict['predictions'].extend(item['predictions'].tolist())

    f = Path(trainer.log_dir) / f'predictions_{trainer.local_rank}.pkl'
    with open(f, 'wb') as fp:
        pickle.dump(out_dict, fp)


def read_predictions(files: Iterable[Path]) -> Dict[str, Dict[str, float]]:
    """Read and combine predictions from .pkl files.

    Args:
        files (Iterable[Path]): All files to read

    Returns:
        Dict[str, Dict[str, float]]: Query IDs mapped to document IDs mapped to scores
    """
    result = defaultdict(dict)
    for f in files:
        with open(f, 'rb') as fp:
            d = pickle.load(fp)
            for q_id, doc_id, prediction in zip(d['q_ids'], d['doc_ids'], d['predictions']):
                result[q_id][doc_id] = prediction
    return result
