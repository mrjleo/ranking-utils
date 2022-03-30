#! /usr/bin/env python3


from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything


@hydra.main(config_path="config", config_name="create_h5_data")
def main(config: DictConfig):
    seed_everything(config.random_seed)
    instantiate(config.dataset).save(
        Path.cwd(), config.num_training_negatives, config.balance_training_pairs
    )


if __name__ == "__main__":
    main()
