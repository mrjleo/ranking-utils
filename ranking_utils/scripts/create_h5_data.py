#! /usr/bin/env python3


from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything


@hydra.main(config_path="config", config_name="create_h5_data", version_base="1.3")
def main(config: DictConfig) -> None:
    seed_everything(config.random_seed)
    instantiate(config.dataset).save(Path.cwd(), **config.training)


if __name__ == "__main__":
    main()
