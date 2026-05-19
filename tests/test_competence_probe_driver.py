"""Driver/CLI tests for the competence probe (Path B A.3).

Covers ONLY the driver wiring: config -> model-id resolution -> orchestrated
N=1 runs over probe seeds -> per-task accuracy -> compute_separation -> JSON
artifact + one-line verdict + unknown-model rejection.

The pure functions (classify_tier / compute_separation) are tested elsewhere;
they are exercised here only end-to-end. No real Ollama: generation is stubbed
via the established pattern (patch src.orchestrator.generate_completions).
"""
import json

import pytest
from unittest.mock import patch

from scripts.competence_probe import main


def _fake_generate_factory(tasks):
    """Build a stub for generate_completions that engineers a KNOWN outcome.

    Easy-tier tasks (category classifies "easy") -> always correct.
    Hard-tier tasks                              -> always incorrect.
    => easy_acc = 1.0, binding_acc = 0.0 => separates True (reason "separates").

    Keyed by prompt because run_single_config calls per-task with task["prompt"].
    """
    by_prompt = {t["prompt"]: t for t in tasks}
    easy_suffixes = ("_easy", "_common", "_simple")

    def fake_generate(prompt, n, model, temperature, seed, **kw):
        task = by_prompt[prompt]
        cat = task["category"]
        is_easy = cat.endswith(easy_suffixes)
        if is_easy:
            ans = task["ground_truth"]
        else:
            ans = "definitely-wrong-answer-xyz"
        return [f"CONFIDENCE: 0.8\nANSWER: {ans}"] * n

    return fake_generate


def test_driver_writes_json_and_separates(params_path, tmp_path):
    """(a) JSON written with expected keys; (b) separates/reason match design."""
    from src.config import load_config
    from src.tasks import load_tasks

    cfg = load_config(params_path)
    tasks = load_tasks(params_path.parent.parent / cfg["tasks"]["file"])
    fake = _fake_generate_factory(tasks)

    out_root = tmp_path / "experiment_output" / "competence_probe"

    with patch("src.orchestrator.generate_completions", side_effect=fake):
        rc = main(["--model", "primary", "--output-root", str(out_root)])

    assert rc == 0
    outfile = out_root / "primary.json"
    assert outfile.exists()

    data = json.loads(outfile.read_text())
    expected_keys = {
        "model", "model_id", "easy_acc", "binding_acc", "h_ceiling",
        "separates", "reason", "per_task_acc",
    }
    assert expected_keys <= set(data)

    assert data["model"] == "primary"
    assert data["model_id"] == cfg["model"]["primary"]  # qwen2.5:7b
    # Designed outcome: easy all correct, hard all wrong.
    assert data["easy_acc"] == pytest.approx(1.0)
    assert data["binding_acc"] == pytest.approx(0.0)
    assert data["separates"] is True
    assert data["reason"] == "separates"
    assert len(data["per_task_acc"]) == cfg["tasks"]["total"]


def test_driver_non_separating_outcome(params_path, tmp_path):
    """Designed FAIL outcome: all tasks wrong -> easy_acc 0 -> not separates."""
    from src.config import load_config
    from src.tasks import load_tasks

    cfg = load_config(params_path)
    tasks = load_tasks(params_path.parent.parent / cfg["tasks"]["file"])

    def fake_all_wrong(prompt, n, model, temperature, seed, **kw):
        return ["CONFIDENCE: 0.8\nANSWER: never-correct-zzz"] * n

    out_root = tmp_path / "co" / "competence_probe"
    with patch("src.orchestrator.generate_completions", side_effect=fake_all_wrong):
        rc = main(["--model", "secondary", "--output-root", str(out_root)])

    assert rc == 0
    data = json.loads((out_root / "secondary.json").read_text())
    assert data["separates"] is False
    assert data["reason"] == "easy_below_min"


def test_driver_unknown_model_rejected(tmp_path):
    """(c) unknown --model exits non-zero / raises; no network touched."""
    out_root = tmp_path / "co"
    with patch("src.orchestrator.generate_completions",
               side_effect=AssertionError("network must not be touched")):
        with pytest.raises(SystemExit) as exc:
            main(["--model", "zzz", "--output-root", str(out_root)])
    assert exc.value.code != 0
