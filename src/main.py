"""
main.py – experiment orchestrator.  Reads `config/config.yaml`, creates the
corresponding `ExperimentConfig` objects and launches training.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

from train import ExperimentConfig, PreTrainer

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------------

def _load_configs(cfg_path: Path) -> List[ExperimentConfig]:
    with cfg_path.open("r") as fp:
        raw = yaml.safe_load(fp)
    # the YAML top-level is expected to be a list under the key "experiments"
    experiments = raw.get("experiments", []) if isinstance(raw, dict) else raw
    return [ExperimentConfig.from_dict(item) for item in experiments]


def main():  # noqa: D401 – imperative entry-point
    parser = argparse.ArgumentParser(description="AutoSlim-LLM Runner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML file with experiment configs")
    parser.add_argument("--spm", type=str, default="sentencepiece_32k.model", help="Path to 32k sentencepiece model")
    args = parser.parse_args()

    cfgs = _load_configs(Path(args.config))
    spm_path = Path(args.spm)

    for cfg in cfgs:
        print("\n============================================================")
        print(f"Running experiment: {cfg.name} (variant={cfg.variant}, seed={cfg.seed})")
        print("============================================================\n")
        _set_seed(cfg.seed)
        trainer = PreTrainer(cfg, tokenizer_path=spm_path)
        metrics = trainer.train()  # noqa: F841 – printed inside


if __name__ == "__main__":
    main()
