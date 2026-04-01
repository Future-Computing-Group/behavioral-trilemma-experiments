#!/usr/bin/env python3
"""Entry point for the behavioral trilemma experiment.

Usage:
    python -m scripts.run --mode unit_smoke
    python -m scripts.run --mode unit_smoke --phase0
    python -m scripts.run --mode full --model secondary
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, generate_configs
from src.tasks import load_tasks
from src.orchestrator import (
    config_filename,
    get_completed_configs,
    run_single_config,
    run_phase0_calibration,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARAMS = ROOT / "configs" / "params.yaml"
PROGRESS_FILE = ROOT / ".progress.json"


def write_progress(status: dict):
    """Write progress to .progress.json (compatible with shared/monitor.sh)."""
    PROGRESS_FILE.write_text(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Behavioral Trilemma Experiment")
    parser.add_argument("--mode", default="unit_smoke",
                        choices=["unit_smoke", "integration_smoke", "full"])
    parser.add_argument("--phase0", action="store_true",
                        help="Run Phase 0 calibration only")
    parser.add_argument("--model", default="primary",
                        choices=["primary", "secondary"])
    parser.add_argument("--config", default=str(PARAMS),
                        help="Path to params.yaml")
    args = parser.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    model_key = cfg["model"][args.model]
    if model_key is None:
        print(f"ERROR: model '{args.model}' is null in config")
        sys.exit(1)

    # Determine task count from mode
    if args.mode in cfg.get("levels", {}):
        n_tasks = cfg["levels"][args.mode].get("tasks", cfg["tasks"]["total"])
    else:
        n_tasks = cfg["tasks"]["total"]

    task_path = ROOT / cfg["tasks"]["file"]
    if not task_path.exists():
        print(f"ERROR: task set not found at {task_path}")
        print("Run: python scripts/generate_tasks.py")
        sys.exit(1)

    tasks = load_tasks(task_path)[:n_tasks]
    results_dir = ROOT / cfg["output"]["results_dir"]
    temperature = cfg["model"]["temperature"]

    print(f"Mode: {args.mode} | Model: {model_key} | Tasks: {len(tasks)}")

    if args.phase0:
        cal_cfg = cfg.get("calibration", {})
        seeds = cal_cfg.get("seeds", list(range(1000, 1020)))
        thresholds = cfg["experiment"]["thresholds"]
        print(f"Phase 0 calibration: {len(tasks)} tasks × {len(seeds)} seeds")

        outfile = run_phase0_calibration(
            tasks, results_dir, model=model_key,
            seeds=seeds, thresholds=thresholds,
            temperature=temperature,
        )
        print(f"Calibration results: {outfile}")
        return

    # Experiment run
    configs = list(generate_configs(cfg, mode=args.mode))
    completed = get_completed_configs(results_dir)
    remaining = [c for c in configs if config_filename(c, model_key) not in completed]

    print(f"Configs: {len(configs)} total, {len(completed)} completed, {len(remaining)} remaining")

    progress = {
        "mode": args.mode,
        "model": model_key,
        "total": len(configs),
        "completed": len(completed),
        "failed": 0,
        "status": "running",
    }
    write_progress(progress)

    for i, c in enumerate(remaining):
        fname = config_filename(c, model_key)
        print(f"[{len(completed)+i+1}/{len(configs)}] {fname}")
        t0 = time.time()

        try:
            run_single_config(c, tasks, results_dir, model=model_key,
                              temperature=temperature)
            progress["completed"] += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            progress["failed"] += 1

        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")
        write_progress(progress)

    progress["status"] = "done"
    write_progress(progress)
    print(f"\nComplete: {progress['completed']} succeeded, {progress['failed']} failed")


if __name__ == "__main__":
    main()
