"""Experiment orchestrator: config sweep, per-config execution, Phase 0 calibration."""
import csv
import pathlib

from src.ollama_client import generate_completions
from src.parser import parse_response
from src.scorer import oracle_payoff, brier_score


def config_filename(cfg: dict, model: str = "qwen2.5:7b") -> str:
    """Generate a deterministic filename for a config's results CSV."""
    m = model.replace(":", "_").replace("/", "_")
    return f"{m}_N{cfg['N']}_w{cfg['w_ratio']}_r{cfg['r_min']}_s{cfg['seed']}.csv"


def get_completed_configs(results_dir: pathlib.Path) -> set[str]:
    """Return set of filenames already completed (for resume support)."""
    if not results_dir.exists():
        return set()
    return {f.name for f in results_dir.glob("*.csv") if f.stat().st_size > 0}


def _verify_answer(task: dict, answer: str | None) -> int:
    """Check if an answer is correct. Returns 1 (correct) or 0 (incorrect)."""
    if answer is None:
        return 0
    gt = task["ground_truth"].strip()
    if task["verification"] == "exact":
        return 1 if answer.strip().lower() == gt.lower() else 0
    elif task["verification"] == "arithmetic":
        # Normalize: strip whitespace, try numeric comparison
        try:
            return 1 if float(answer.strip()) == float(gt) else 0
        except ValueError:
            return 1 if answer.strip() == gt else 0
    elif task["verification"] == "code":
        # Run the model's code against the task's test cases.
        # If any test case fails, the answer is incorrect.
        test_cases = task.get("test_cases", [])
        if not test_cases:
            return 0
        try:
            from src.tasks import verify_code
            return 1 if verify_code(answer.strip(), test_cases) else 0
        except Exception:
            return 0
    return 0


def run_single_config(
    cfg: dict,
    tasks: list[dict],
    results_dir: pathlib.Path,
    model: str = "qwen2.5:7b",
    temperature: float = 0.8,
) -> pathlib.Path:
    """Run one config across all tasks. Write per-task results to CSV.

    For each task: generate N completions, score by oracle payoff, select argmax.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    outfile = results_dir / config_filename(cfg, model)

    rows = []
    for task in tasks:
        completions = generate_completions(
            prompt=task["prompt"],
            n=cfg["N"],
            model=model,
            temperature=temperature,
            seed=cfg["seed"],
        )

        # Parse and score each completion
        best_v = float("-inf")
        best_row = None

        for comp in completions:
            r, a = parse_response(comp)
            if r is None:
                continue  # Skip unparseable completions

            y = _verify_answer(task, a)
            v = oracle_payoff(r, y, cfg["w_C"], cfg["w_A"], cfg["r_min"])

            if v > best_v:
                best_v = v
                best_row = {
                    "task_id": task["id"],
                    "category": task["category"],
                    "N": cfg["N"],
                    "w_ratio": cfg["w_ratio"],
                    "w_C": cfg["w_C"],
                    "w_A": cfg["w_A"],
                    "r_min": cfg["r_min"],
                    "seed": cfg["seed"],
                    "r_selected": r,
                    "y": y,
                    "V_selected": v,
                    "brier": brier_score(r, y),
                    "gate_cleared": 1 if r >= cfg["r_min"] else 0,
                }

        if best_row is None:
            # All completions unparseable — record as missing (L39: don't silently skip)
            best_row = {
                "task_id": task["id"],
                "category": task["category"],
                "N": cfg["N"],
                "w_ratio": cfg["w_ratio"],
                "w_C": cfg["w_C"],
                "w_A": cfg["w_A"],
                "r_min": cfg["r_min"],
                "seed": cfg["seed"],
                "r_selected": "",
                "y": "",
                "V_selected": "",
                "brier": "",
                "gate_cleared": "",
            }
        rows.append(best_row)

    # Write CSV
    fieldnames = [
        "task_id", "category", "N", "w_ratio", "w_C", "w_A", "r_min",
        "seed", "r_selected", "y", "V_selected", "brier", "gate_cleared",
    ]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return outfile


def run_phase0_calibration(
    tasks: list[dict],
    results_dir: pathlib.Path,
    model: str = "qwen2.5:7b",
    seeds: list[int] | None = None,
    thresholds: list[float] | None = None,
    temperature: float = 0.8,
) -> pathlib.Path:
    """Run Phase 0: estimate p_hat per task using held-out seeds.

    For each task, run len(seeds) times (N=1, w_A=0), check correctness,
    compute p_hat = fraction correct. Classify binding sets per threshold.
    """
    if seeds is None:
        seeds = list(range(1000, 1020))
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.9]

    results_dir.mkdir(parents=True, exist_ok=True)
    outfile = results_dir / "phase0_calibration.csv"

    rows = []
    for task in tasks:
        correct_count = 0
        for seed in seeds:
            completions = generate_completions(
                prompt=task["prompt"],
                n=1,
                model=model,
                temperature=temperature,
                seed=seed,
            )
            if completions:
                _, a = parse_response(completions[0])
                correct_count += _verify_answer(task, a)

        p_hat = correct_count / len(seeds) if seeds else 0.0

        row = {"task_id": task["id"], "p_hat": round(p_hat, 4)}
        for t in thresholds:
            row[f"binding_{t}"] = 1 if p_hat < t else 0
        rows.append(row)

    fieldnames = ["task_id", "p_hat"] + [f"binding_{t}" for t in thresholds]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return outfile
