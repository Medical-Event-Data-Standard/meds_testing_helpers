#!/usr/bin/env python

import logging

from . import CONFIG_YAML

logger = logging.getLogger(__name__)

import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Generate a dataset of the specified size."""

    cfg = hydra.utils.instantiate(cfg)

    output_dir = Path(cfg.output_dir)

    if output_dir.exists():
        if output_dir.is_file():
            raise ValueError("Output directory is a file; expected a directory.")
        if cfg.do_overwrite:
            logger.warning("Output directory already exists. Overwriting.")
            shutil.rmtree(output_dir)
        elif (output_dir / "data").exists() or (output_dir / "metadata").exists():
            contents = [f"  - {p.relative_to(output_dir)}" for p in output_dir.rglob("*")]
            contents_str = "\n".join(contents)
            raise ValueError(
                f"Output directory is not empty! use --do-overwrite to overwrite. Contents:\n{contents_str}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    G = hydra.utils.instantiate(cfg.dataset_spec)
    rng = np.random.default_rng(cfg.seed)

    logger.info(f"Generating dataset with {cfg.N_subjects} subjects.")
    dataset = G.sample(cfg.N_subjects, rng)

    logger.info(f"Saving dataset to root directory {str(output_dir.resolve())}.")
    dataset.write(output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
