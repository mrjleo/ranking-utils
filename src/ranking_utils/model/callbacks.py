from pathlib import Path
from typing import Sequence, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from ranking_utils.model import Ranker


class RankingPredictionWriter(BasePredictionWriter):
    def __init__(self, out_dir: Union[Path, str]) -> None:
        super().__init__("epoch")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: Ranker,
        predictions: Sequence[torch.Tensor],
        batch_indices: Sequence[int],
    ) -> None:
        # include the rank in the file name, otherwise multiple processes compete with each other
        out_file = self.out_dir / f"out_{trainer.global_rank}"
        torch.save(predictions, out_file)
