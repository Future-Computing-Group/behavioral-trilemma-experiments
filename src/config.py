"""Config loader for behavioral trilemma experiments."""
import pathlib
from itertools import product
from typing import Any

import yaml


def load_config(path: pathlib.Path) -> dict[str, Any]:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def generate_configs(cfg: dict, mode: str = "full"):
    """Yield per-run config dicts for the given mode.

    Each config contains: N, w_ratio, w_C, w_A, r_min, seed.
    """
    exp = cfg["experiment"]
    w_C = exp["w_C"]

    if mode in cfg.get("levels", {}):
        level = cfg["levels"][mode]
        N_values = level.get("N_values", exp["N_values"])
        weight_ratios = level.get("weight_ratios", exp["weight_ratios"])
        thresholds = level.get("thresholds", exp["thresholds"])
        seeds = level.get("seeds", exp["seeds"])
    else:
        N_values = exp["N_values"]
        weight_ratios = exp["weight_ratios"]
        thresholds = exp["thresholds"]
        seeds = exp["seeds"]

    # Normalize seeds to list (YAML may parse single int)
    if isinstance(seeds, int):
        seeds = [seeds]

    for N, w_ratio, r_min, seed in product(N_values, weight_ratios, thresholds, seeds):
        yield {
            "N": N,
            "w_ratio": w_ratio,
            "w_C": w_C,
            "w_A": w_C * w_ratio,
            "r_min": r_min,
            "seed": seed,
        }
