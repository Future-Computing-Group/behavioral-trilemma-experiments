#!/usr/bin/env python3
"""Stage-A pool generator: produce ``pool_qwen2.5_7b_seed{S}_N32.jsonl`` files.

Background. The behavioral-trilemma 540-config sweep behind §7 Table 1 was a
two-stage pipeline. Stage A (this module) generates a pool of 32 completions
per (task, seed) by calling Ollama's OpenAI-compatible logprob endpoint, and
captures per-completion {raw, parsed verbalized confidence, logprob-confidence
geomean, parsed answer, oracle y}. Stage B
(``scripts/select_from_pool.py``) consumes the pool and emits the 540
per-config selected-task CSVs that ``analysis/hypothesis_tests.py`` reads.

The original Stage A lived only on a coauthor's local machine
(commit b453d6d 2026-04-20 added the pool jsonls + per-config CSVs but never
the generator). The audit (B-A1-1) flagged this as a reproducibility blocker
for the manuscript's headline data; this module is the reconstruction. It is
NOT expected to produce byte-identical pools to the original — Ollama
output is hardware/build-dependent at non-zero temperature, per the
manuscript's own §exp-compute statement that "re-runs reproduce the reported
effects rather than byte-identical records." Statistical equivalence is the
contract.

Per-completion seeding. ``src/ollama_client.py:38-40`` documents the
codebase's established convention: the i-th completion at base-seed S uses
Ollama seed ``S + i``. We mirror that here. If the original Stage A used a
different per-completion seed schedule, the reconstructed pool's
per-completion records will differ from the shipped pool record-by-record
(but should reproduce the hypothesis decisions within stochastic tolerance).
A note describing the inferred schedule is written to ``pool_meta_*.json``.

Usage
-----
::

    # smoke: 1 task × 1 seed × 4 completions (~30s, real Ollama call)
    python -m scripts.generate_pool --smoke

    # one seed (100 tasks × 32 completions, ~2.5h on a Mac Studio)
    python -m scripts.generate_pool --seed 0

    # full sweep (5 seeds, ~14h)
    python -m scripts.generate_pool --all-seeds

See ``tests/test_generate_pool.py`` for the (network-free) unit contract.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.logprob_confidence import logprob_confidence  # noqa: E402
from src.ollama_logprob_client import generate_with_logprobs  # noqa: E402
from src.orchestrator import _verify_answer  # noqa: E402
from src.parser import parse_response  # noqa: E402
from src.status import write_status  # noqa: E402
from src.tasks import load_tasks  # noqa: E402

# Pool record / completion schemas — verified against the shipped pool
# (``experiment_output/raw_runs/logprob/pools/pool_qwen2.5_7b_seed0_N32.jsonl``).
POOL_RECORD_FIELDS = [
    "task_id", "category", "seed", "confidence_source", "completions",
]
POOL_COMPLETION_FIELDS = [
    "idx", "raw", "r", "r_verbalized", "r_logprob", "mean_logprob", "a", "y",
]

# Defaults pinned by ``configs/params.yaml`` and the shipped pool_meta JSON.
N_MAX = 32
TEMPERATURE = 0.8
MAX_TOKENS = 512
MODEL = "qwen2.5:7b"
MODEL_SLUG = "qwen2.5_7b"
SEEDS = [42, 123, 456, 789, 0]

POOLS_DIR = ROOT / "experiment_output" / "raw_runs" / "logprob" / "pools"
DEFAULT_TASK_SET = ROOT / "tasks" / "task_set.json"


# ---------------------------------------------------------------------------
# Pure record-builders (network-free; mockable in tests)
# ---------------------------------------------------------------------------

def build_completion_record(idx: int, completion, y: int) -> dict:
    """Convert one ``LogprobCompletion`` + its oracle y into a pool-record
    entry. ``completion`` is duck-typed to expose ``content`` and
    ``token_logprobs`` (so the test suite can pass a fake without hitting
    Ollama).

    Field semantics, locked to the shipped pool record schema:

    * ``raw`` — verbatim model output
    * ``r_verbalized`` — confidence parsed from the model's
      ``CONFIDENCE: …`` line via ``src/parser.py``; ``None`` if absent.
    * ``r_logprob`` — geometric mean of per-token probabilities (the
      manuscript's confidence equation; computed by
      ``analysis.logprob_confidence``).
    * ``r`` — the canonical confidence; since ``confidence_source="logprob"``
      for this pool, ``r == r_logprob``.
    * ``mean_logprob`` — mean of token logprobs (i.e., ``ln(r_logprob)``
      pre-clip); preserved verbatim for downstream diagnostics.
    * ``a`` — answer parsed from the model's ``ANSWER: …`` line; ``None``
      if no answer was parsed.
    * ``y`` — oracle correctness, passed in by the caller (this function
      does NOT verify, so its purity stays trivial).
    """
    r_verbalized, a = parse_response(completion.content)
    r_logprob = logprob_confidence(completion.token_logprobs)
    mean_logprob = sum(completion.token_logprobs) / len(completion.token_logprobs)
    return {
        "idx": idx,
        "raw": completion.content,
        "r": r_logprob,           # canonical for the logprob path
        "r_verbalized": r_verbalized,
        "r_logprob": r_logprob,
        "mean_logprob": mean_logprob,
        "a": a,
        "y": y,
    }


def build_pool_record(task: dict, seed: int, completions: list, verify_y) -> dict:
    """Assemble one pool record from N completions for one (task, seed).

    ``verify_y`` is a function ``(task, answer) -> 0|1`` injected so the
    real verifier (``src.orchestrator._verify_answer``) can be swapped with
    a stub in tests.
    """
    comp_records = []
    for i, comp in enumerate(completions):
        _r_verb, a = parse_response(comp.content)
        y = verify_y(task, a)
        comp_records.append(build_completion_record(idx=i, completion=comp, y=y))
    return {
        "task_id": task["id"],
        "category": task["category"],
        "seed": seed,
        "confidence_source": "logprob",
        "completions": comp_records,
    }


# ---------------------------------------------------------------------------
# Driver (network-using; the part that calls Ollama)
# ---------------------------------------------------------------------------

def _per_completion_seed(base_seed: int, i: int) -> int:
    """Per-completion seed schedule, inferred from src/ollama_client.py:38-40
    (the codebase's established convention: i-th draw at base seed S uses
    Ollama seed S+i). If the original Stage A used a different schedule,
    the regenerated pool's per-completion records will differ from the
    shipped pool record-by-record but should reproduce the hypothesis
    decisions within stochastic tolerance."""
    return base_seed + i


def generate_pool_record_for_task(task: dict, seed: int, n_max: int = N_MAX,
                                  model: str = MODEL,
                                  temperature: float = TEMPERATURE,
                                  max_tokens: int = MAX_TOKENS) -> dict:
    """Network-using: call Ollama N times for one (task, seed) and assemble
    a pool record. The unit tests don't exercise this function (they go
    through ``build_*`` directly with mocks); the ``--smoke`` driver flag
    exercises it for a single cell against real Ollama.
    """
    completions = []
    for i in range(n_max):
        comp = generate_with_logprobs(
            prompt=task["prompt"],
            model=model,
            temperature=temperature,
            seed=_per_completion_seed(seed, i),
            max_tokens=max_tokens,
        )
        completions.append(comp)
    return build_pool_record(
        task=task, seed=seed, completions=completions,
        verify_y=_verify_answer,
    )


def model_slug(model: str) -> str:
    """Filesystem slug for an Ollama model tag. Identical rule to
    ``scripts.eval_logprob._model_slug`` (kept in sync by test)."""
    return model.replace(":", "_").replace("/", "_")


def pool_path(seed: int, pools_dir: pathlib.Path = POOLS_DIR,
              model_slug: str = MODEL_SLUG) -> pathlib.Path:
    return pools_dir / f"pool_{model_slug}_seed{seed}_N{N_MAX}.jsonl"


def pool_meta_path(pools_dir: pathlib.Path = POOLS_DIR,
                   model_slug: str = MODEL_SLUG) -> pathlib.Path:
    return pools_dir / f"pool_meta_{model_slug}_N{N_MAX}.json"


def write_pool_meta(pools_dir: pathlib.Path = POOLS_DIR,
                    model: str = MODEL,
                    model_slug: str = MODEL_SLUG) -> pathlib.Path:
    pools_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": model,
        "n_max": N_MAX,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "task_count": 100,
        "seeds": SEEDS,
        "confidence_source": "logprob",
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_completion_seed_schedule": "base_seed + i",
        "note": (
            "Generated by scripts/generate_pool.py — Stage-A reconstruction "
            "per audit B-A1-1. Per-completion seed schedule inferred from "
            "src/ollama_client.py convention; not guaranteed identical to "
            "the original (uncommitted) Stage A. Statistical equivalence, "
            "not byte-equality, is the contract — see the module docstring."
        ),
    }
    path = pool_meta_path(pools_dir, model_slug=model_slug)
    path.write_text(json.dumps(meta, indent=2))
    return path


def _salvage_completed_task_ids(out_path: pathlib.Path) -> set[str]:
    """L69 resume support: read completed task_ids from an existing pool
    file, dropping a trailing partial line (crash mid-write) by atomically
    rewriting the file without it (tmp + ``os.replace``). Returns the set
    of task_ids whose records are intact on disk."""
    if not out_path.exists():
        return set()
    raw_lines = out_path.read_text().splitlines()
    good_lines: list[str] = []
    completed: set[str] = set()
    salvage_needed = False
    for line in raw_lines:
        if not line.strip():
            salvage_needed = True       # drop blank padding too
            continue
        try:
            rec = json.loads(line)
            completed.add(rec["task_id"])
            good_lines.append(line)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Only a trailing partial line is expected; anything after it
            # is unreachable in the append-only write pattern, but drop
            # defensively either way.
            salvage_needed = True
    if salvage_needed:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text("".join(line + "\n" for line in good_lines))
        os.replace(tmp, out_path)
    return completed


def write_pool_for_seed(seed: int, tasks: list[dict],
                        pools_dir: pathlib.Path = POOLS_DIR,
                        n_max: int = N_MAX,
                        progress: bool = True,
                        model: str = MODEL,
                        model_slug: str = MODEL_SLUG) -> pathlib.Path:
    """Generate and write one seed's worth (100 tasks × n_max completions).
    Wall time is ~2-3 h on a Mac Studio with qwen2.5:7b warm.

    L69: resumable + idempotent. Completed (task, seed) records on disk are
    byte-preserved and skipped (append mode, never ``"w"``); a trailing
    partial line from a crash is salvaged out before appending.
    """
    pools_dir.mkdir(parents=True, exist_ok=True)
    out_path = pool_path(seed, pools_dir, model_slug=model_slug)
    # L63: STATUS.json at the per-model output root (§11.E-B.5), i.e. the
    # parent of the pools dir; unconditional per-task emission.
    status_path = pools_dir.parent / "STATUS.json"
    completed = _salvage_completed_task_ids(out_path)
    completions_done = 0
    parse_fail_count = 0
    t0 = time.time()
    with out_path.open("a") as f:
        for j, task in enumerate(tasks):
            if task["id"] in completed:
                continue
            t_task = time.time()
            rec = generate_pool_record_for_task(task=task, seed=seed,
                                                n_max=n_max, model=model)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            completions_done += len(rec["completions"])
            parse_fail_count += sum(
                1 for c in rec["completions"] if c["a"] is None)
            dt = time.time() - t_task
            total = time.time() - t0
            eta = (total / (j + 1)) * (len(tasks) - j - 1)
            write_status(status_path, stage="stageA", model=model,
                         seed=seed, task_idx=j + 1, total_tasks=len(tasks),
                         completions_done=completions_done,
                         parse_fail_count=parse_fail_count,
                         elapsed_s=round(total, 1), eta_s=round(eta, 1))
            if progress:
                print(
                    f"[seed={seed}] task {j+1}/{len(tasks)} {task['id']}: "
                    f"{n_max} completions in {dt:.1f}s "
                    f"(elapsed {total/60:.1f}m, eta {eta/60:.1f}m)",
                    flush=True,
                )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="generate_pool",
        description="Stage-A pool generator (audit B-A1-1 reconstruction).",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Generate 4 completions for 1 task at seed=0 against "
                        "real Ollama (~30 s; verifies wiring end-to-end).")
    p.add_argument("--seed", type=int,
                   help="Generate one seed's pool (~2-3 h).")
    p.add_argument("--all-seeds", action="store_true",
                   help="Generate all 5 seeds (~14 h).")
    p.add_argument("--task-set", type=pathlib.Path, default=DEFAULT_TASK_SET)
    p.add_argument("--pools-dir", type=pathlib.Path, default=None,
                   help="Pool output dir. Default: the legacy logprob/pools "
                        "dir for the default model, "
                        "experiment_output/raw_runs/logprob-<slug>/pools/ "
                        "for any other --model (§11.E-B.5).")
    p.add_argument("--n-max", type=int, default=N_MAX)
    p.add_argument("--n-tasks", type=int, default=None,
                   help="Subset the task set to the first N tasks (smoke).")
    p.add_argument("--model", default=MODEL,
                   help=f"Ollama model tag (default: {MODEL}). Output "
                        "filenames use the derived slug.")
    args = p.parse_args(argv)

    chosen = sum([bool(args.smoke), args.seed is not None, args.all_seeds])
    if chosen != 1:
        p.error("exactly one of --smoke / --seed / --all-seeds is required")

    slug = model_slug(args.model)
    if args.pools_dir is None:
        if args.model == MODEL:
            args.pools_dir = POOLS_DIR          # legacy path, byte-stable
        else:
            args.pools_dir = (ROOT / "experiment_output" / "raw_runs"
                              / f"logprob-{slug}" / "pools")

    args.pools_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.task_set)
    # Sort alphabetically by task id so the pool jsonl matches the order
    # downstream Stage B expects (Stage B sorts on read, so this matters
    # for human inspection but not for correctness).
    tasks.sort(key=lambda t: t["id"])
    if args.n_tasks is not None:
        tasks = tasks[: args.n_tasks]

    if args.smoke:
        task = tasks[0]
        print(f"[smoke] generating 4 completions for {task['id']} seed=0 ...")
        rec = generate_pool_record_for_task(
            task=task, seed=0, n_max=4, model=args.model,
        )
        print(json.dumps({
            "task_id": rec["task_id"],
            "n_completions": len(rec["completions"]),
            "first_completion": rec["completions"][0],
        }, indent=2)[:800])
        # Stat-non-degeneracy check: at least 2 distinct r values.
        rs = sorted({c["r"] for c in rec["completions"]})
        if len(rs) < 2:
            print(
                f"WARNING: only {len(rs)} distinct r value(s) across "
                f"{len(rec['completions'])} completions: {rs!r}; check that "
                "per-completion seeds are actually varying.",
                file=sys.stderr,
            )
            return 1
        print(f"distinct r values: {len(rs)}; range "
              f"[{min(rs):.4f}, {max(rs):.4f}] -- non-degenerate OK")
        return 0

    write_pool_meta(args.pools_dir, model=args.model, model_slug=slug)
    seeds = [args.seed] if args.seed is not None else SEEDS
    for s in seeds:
        out_path = write_pool_for_seed(s, tasks, args.pools_dir,
                                       n_max=args.n_max, model=args.model,
                                       model_slug=slug)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
