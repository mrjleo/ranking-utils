#! /usr/bin/env python3


from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything


@hydra.main(config_path="config", config_name="create_h5_data")
def main(config: DictConfig) -> None:
    seed_everything(config.random_seed)
    instantiate(config.dataset).save(
        Path.cwd(),
        config.training.num_instances_per_positive,
        config.training.balance_queries,
        config.training.balance_labels,
    )


if __name__ == "__main__":
    main()
