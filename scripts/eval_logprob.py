#!/usr/bin/env python3
"""Run ONE model through the logprob-confidence path on the 100-task set.

Operational glue for the logprob cross-model figure (L.3). For one model
and one seed, this calls ``src.ollama_logprob_client.generate_with_logprobs``
once per task and writes per-output rows to::

    experiment_output/logprob_xmodel/<model>/<model>_s<seed>.csv

with columns EXACTLY (schema-comparable to the competence-probe CSVs)::

    task,category,seed,r_logprob,answer,y,acted

Column semantics
----------------
* ``task``      — task["id"] (verbatim).
* ``category``  — task["category"] (verbatim; mirrors src/orchestrator.py
  exactly so logprob CSVs are schema-comparable to competence-probe CSVs).
* ``seed``      — the run seed.
* ``r_logprob`` — ``LogprobCompletion.confidence``: the logprob-confidence.
  It is ALWAYS present — every completion has tokens, so the token-logprob
  confidence always exists.
* ``answer``    — the answer parsed from ``LogprobCompletion.content`` via
  ``src.parser.parse_response`` (the existing ANSWER parser); empty string
  if no answer was parsed.
* ``acted``     — **PINNED, load-bearing (v2-IMPORTANT; do not let drift).**
  ``acted = 1`` iff a valid answer was successfully parsed from the
  completion via ``src/parser.py``'s ANSWER parser; ``acted = 0``
  (abstained) otherwise. This is DELIBERATELY answer-parse-based. It is NOT
  the archived competence-probe predicate (``r_selected`` non-blank, i.e.
  confidence-parseable post-argmax). The logprob path has no argmax (N=1)
  and ``r_logprob`` is ALWAYS present from the token logprobs, so an
  ``r_logprob``-non-blank predicate would make EVERY row "acted" — wrong.
  The manuscript autonomy A is the answer-commitment / non-abstention rate
  (see main.tex:968 autonomy definition; also consistent with
  ``analysis.model_points.HONEST_CAPTION``'s "answer-commitment rate"
  framing). Answer-parse ``acted`` matches that; confidence-parse would not.
* ``y``         — 1 if the parsed answer equals the task's gold answer,
  else 0. Computed via ``src.orchestrator._verify_answer`` (the SAME
  correctness function the competence probe uses through
  ``run_single_config``), so ``y`` is byte-for-byte schema-parity with the
  existing competence-probe CSVs. Only meaningful for acted rows; for
  abstained rows ``answer`` is empty so ``_verify_answer`` returns 0 and
  ``y`` is written as 0 (the loader excludes abstained rows from C, counts
  them as not-acted for A, and contributes H=0 for that task).

This script is OPERATIONAL GLUE. Unit tests MUST mock
``generate_with_logprobs`` (no Ollama / no network). Real model runs are a
later task; this script does not run real models on import or in tests.
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ollama_logprob_client import generate_with_logprobs
from src.orchestrator import _verify_answer
from src.parser import parse_response
from src.tasks import load_tasks

# Canonical 100-task set + cross-model output root (sensible defaults).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_TASK_SET = _REPO_ROOT / "tasks" / "task_set.json"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "experiment_output" / "logprob_xmodel"

_FIELDNAMES = ["task", "category", "seed", "r_logprob", "answer", "y", "acted"]


def _model_slug(model: str) -> str:
    """Filesystem-safe model id (mirror orchestrator.config_filename)."""
    return model.replace(":", "_").replace("/", "_")


def run_eval(
    model: str,
    seed: int,
    task_set_path: str,
    output_dir: str,
    temperature: float = 0.8,
) -> str:
    """Run ``model`` through the logprob path on the task set for ``seed``.

    Generation is delegated to ``generate_with_logprobs`` (mockable in
    tests). Returns the written CSV path as a string.
    """
    tasks = load_tasks(pathlib.Path(task_set_path))

    slug = _model_slug(model)
    out_dir = pathlib.Path(output_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{slug}_s{seed}.csv"

    rows = []
    for task in tasks:
        comp = generate_with_logprobs(
            prompt=task["prompt"],
            model=model,
            temperature=temperature,
            seed=seed,
        )
        # r_logprob is the logprob-confidence — ALWAYS present (the
        # completion always has tokens). NOT used for the acted predicate.
        r_logprob = comp.confidence

        # acted := a valid ANSWER was parsed from the completion content,
        # via src/parser.py's existing ANSWER parser (the v2-IMPORTANT
        # pin). parse_response returns (confidence, answer); we use ONLY
        # the answer side here — confidence-parse is deliberately not the
        # acted predicate.
        _r_ignored, answer = parse_response(comp.content)
        acted = 1 if answer is not None else 0

        # y via the SAME correctness function the competence probe uses
        # (src.orchestrator._verify_answer), for exact schema parity.
        # _verify_answer(answer=None) -> 0, so abstained rows get y=0.
        y = _verify_answer(task, answer)

        rows.append({
            "task": task["id"],
            "category": task["category"],
            "seed": seed,
            "r_logprob": r_logprob,
            "answer": answer if answer is not None else "",
            "y": y,
            "acted": acted,
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return str(out_csv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_logprob",
        description="Run one model through the logprob-confidence path on "
                    "the 100-task set for one seed.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Ollama model tag (e.g. qwen2.5:7b).",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Generation seed.",
    )
    parser.add_argument(
        "--task-set", default=str(_DEFAULT_TASK_SET),
        help=f"Path to the 100-task JSON (default: {_DEFAULT_TASK_SET}).",
    )
    parser.add_argument(
        "--output-dir", default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Output root (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    args = parser.parse_args(argv)

    out_csv = run_eval(
        model=args.model,
        seed=args.seed,
        task_set_path=args.task_set,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
