"""Multi-model competence probe (Path B A.2 pure fns + A.3 driver).

The pure functions (`classify_tier`, `compute_separation`) take no I/O and
read no config. The `main()` driver wires config + tasks + the existing
orchestrator generation path into a per-model separation verdict + JSON
artifact. Generation is delegated to `src.orchestrator.run_single_config`
(reused, not reimplemented) so tests can stub Ollama via the established
`patch("src.orchestrator.generate_completions")` pattern.
"""
import argparse
import csv
import json
import pathlib
import sys

_EASY_SUFFIXES = ("_easy", "_common", "_simple")
_HARD_SUFFIXES = ("_hard", "_obscure", "_algorithmic")


def classify_tier(category: str) -> str:
    """Return "easy" or "hard" by the fixed suffix rule.

    Unknown suffix fails loud (no default bucket).
    """
    if category.endswith(_EASY_SUFFIXES):
        return "easy"
    if category.endswith(_HARD_SUFFIXES):
        return "hard"
    raise ValueError(f"unknown category suffix: {category!r}")


def _tier_mean(per_task_acc, task_category, tier):
    """Mean accuracy over tasks whose category classifies as `tier`.

    Empty tier set fails loud.
    """
    vals = [
        per_task_acc[tid]
        for tid, cat in task_category.items()
        if classify_tier(cat) == tier
    ]
    if not vals:
        raise ValueError(f"empty task set for tier: {tier!r}")
    return sum(vals) / len(vals)


def compute_separation(
    per_task_acc: dict[str, float],
    task_category: dict[str, str],
    easy_min_acc: float = 0.70,
    binding_max_acc: float = 0.45,
    min_spread: float = 0.20,
) -> dict:
    """Evaluate the three-clause separation criterion.

    Clauses are independent and evaluated in order 1 -> 2 -> 3; `reason`
    is the first failing clause's tag, or "separates" if all pass.
    """
    easy_acc = _tier_mean(per_task_acc, task_category, "easy")
    binding_acc = _tier_mean(per_task_acc, task_category, "hard")

    clause1 = easy_acc >= easy_min_acc
    clause2 = binding_acc <= binding_max_acc
    clause3 = (easy_acc - binding_acc) >= min_spread

    if not clause1:
        reason = "easy_below_min"
    elif not clause2:
        reason = "no_binding_tasks"
    elif not clause3:
        reason = "insufficient_spread"
    else:
        reason = "separates"

    return {
        "easy_acc": easy_acc,
        "binding_acc": binding_acc,
        "h_ceiling": easy_acc,
        "separates": clause1 and clause2 and clause3,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Driver (Path B A.3): CLI entry that runs the probe for one model.
# --------------------------------------------------------------------------- #

_DEFAULT_OUTPUT_ROOT = "experiment_output/competence_probe"


def _probe_per_task_acc(tasks, results_dir, model_id, seeds):
    """Per-task accuracy over the probe seeds, reusing the orchestrator.

    For each probe seed, run the existing single-config path at N=1 with the
    annotation weight off (w_A=0): this exercises the real generation +
    parsing + scoring code (no new generation loop). The per-task `y` column
    (1 correct / 0 incorrect / "" missing) is averaged across seeds.
    """
    # Imported here so tests patching src.orchestrator.generate_completions
    # bind to the same module object the orchestrator uses.
    from src.orchestrator import run_single_config

    correct = {t["id"]: 0.0 for t in tasks}
    counted = {t["id"]: 0 for t in tasks}

    for seed in seeds:
        cfg = {
            "N": 1,
            "w_ratio": 0,
            "w_C": 1.0,
            "w_A": 0.0,
            "r_min": 0.5,
            "seed": seed,
        }
        outfile = run_single_config(cfg, tasks, results_dir, model=model_id)
        with open(outfile) as f:
            for row in csv.DictReader(f):
                y = row["y"]
                if y == "":
                    continue  # unparseable completion: don't count this run
                correct[row["task_id"]] += float(y)
                counted[row["task_id"]] += 1

    return {
        tid: (correct[tid] / counted[tid] if counted[tid] else 0.0)
        for tid in correct
    }


def main(argv=None) -> int:
    """`python -m scripts.competence_probe --model <key>` driver.

    Returns 0 on success; raises SystemExit (non-zero) on an unknown model
    key (via argparse error semantics).
    """
    parser = argparse.ArgumentParser(
        prog="competence_probe",
        description="Multi-model competence probe: easy vs binding separation.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model registry key under cfg['model'] (e.g. primary, mistral7b).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to params.yaml (default: <repo>/configs/params.yaml).",
    )
    parser.add_argument(
        "--output-root", default=None,
        help=f"Output dir for <model>.json (default: {_DEFAULT_OUTPUT_ROOT}).",
    )
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    config_path = (
        pathlib.Path(args.config) if args.config
        else repo_root / "configs" / "params.yaml"
    )

    from src.config import load_config
    from src.tasks import load_tasks

    cfg = load_config(config_path)

    model_registry = cfg["model"]
    if args.model not in model_registry:
        # Reject unknown key with a clear, non-zero exit (SystemExit).
        known = ", ".join(
            k for k, v in model_registry.items() if isinstance(v, str)
        )
        parser.error(
            f"unknown model key {args.model!r}; known keys: {known}"
        )
    model_id = model_registry[args.model]

    tasks_path = repo_root / cfg["tasks"]["file"]
    tasks = load_tasks(tasks_path)

    probe = cfg["competence_probe"]
    seeds = probe["seeds"]
    sep_cfg = probe["separation"]

    out_root = (
        pathlib.Path(args.output_root) if args.output_root
        else repo_root / _DEFAULT_OUTPUT_ROOT
    )
    results_dir = out_root / "_runs"

    per_task_acc = _probe_per_task_acc(tasks, results_dir, model_id, seeds)
    task_category = {t["id"]: t["category"] for t in tasks}

    sep = compute_separation(
        per_task_acc,
        task_category,
        easy_min_acc=sep_cfg["easy_min_acc"],
        binding_max_acc=sep_cfg["binding_max_acc"],
    )

    out_root.mkdir(parents=True, exist_ok=True)
    outfile = out_root / f"{args.model}.json"
    payload = {
        "model": args.model,
        "model_id": model_id,
        "easy_acc": sep["easy_acc"],
        "binding_acc": sep["binding_acc"],
        "h_ceiling": sep["h_ceiling"],
        "separates": sep["separates"],
        "reason": sep["reason"],
        "per_task_acc": per_task_acc,
    }
    outfile.write_text(json.dumps(payload, indent=2))

    print(
        f"{args.model}: separates={sep['separates']} "
        f"(reason={sep['reason']}; "
        f"easy={sep['easy_acc']:.3f} binding={sep['binding_acc']:.3f})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
