"""Tests for the logprob cross-model loader + eval_logprob script.

TDD RED step: written before the implementation exists. All fixtures use
tmp_path and synthetic CSVs; the real experiment_output/ is never touched.

The load-bearing pin under test (v2-IMPORTANT): the logprob-path ``acted``
predicate is ANSWER-parse-based (a valid answer was parsed from the
completion via src/parser.py), NOT the archived competence-probe predicate
(``r_selected`` non-blank). Because every logprob completion has tokens,
``r_logprob`` (the logprob-confidence) is ALWAYS present, so an
``r_logprob``-non-blank predicate would mark every row "acted" — wrong.
The manuscript autonomy A is the answer-commitment / non-abstention rate
(see main.tex:968 autonomy definition; consistent with
model_points.HONEST_CAPTION's "answer-commitment rate" framing).
"""

import csv
import json
import math
import pathlib
from unittest.mock import patch

import pytest

from analysis.model_points import (
    HONEST_CAPTION,
    HONEST_CAPTION_LOGPROB,
    logprob_model_coords,
    all_logprob_model_coords,
)

LOGPROB_HEADER = ["task", "category", "seed", "r_logprob", "answer", "y", "acted"]


def _write_logprob_csv(path: pathlib.Path, rows: list[dict]) -> str:
    """Write a logprob_xmodel CSV with the canonical 7-column header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOGPROB_HEADER)
        w.writeheader()
        for i, r in enumerate(rows):
            full = {
                "task": r.get("task", f"t{i:03d}"),
                "category": r.get("category", "arithmetic_easy"),
                "seed": r.get("seed", 2000),
                "r_logprob": r.get("r_logprob", ""),
                "answer": r.get("answer", ""),
                "y": r.get("y", ""),
                "acted": r.get("acted", 0),
            }
            w.writerow(full)
    return str(path)


def _row(task, r_logprob, answer, y, acted):
    return {"task": task, "r_logprob": r_logprob, "answer": answer,
            "y": y, "acted": acted}


# ===== 1. acted predicate is ANSWER-parse-based (the v2-IMPORTANT pin) =====

def test_acted_is_answer_parse_not_r_logprob_blank(tmp_path):
    """Rows with a non-blank r_logprob but NO parsed answer MUST be
    acted==0 and excluded from A's numerator.

    main.tex:968 defines autonomy as the answer-commitment / non-abstention
    rate; A here MUST follow the answer-parse acted flag, NOT an
    r_logprob-non-blank predicate (which would mark every logprob row
    "acted" because token-logprob confidence is ALWAYS present).
    """
    # 4 tasks. ALL have a non-blank r_logprob (always true in the logprob
    # path). Only 2 produced a parseable answer (acted==1); the other 2
    # produced confidence-bearing tokens but no ANSWER: line (acted==0).
    rows = [
        _row("t0", 0.90, "788", 1, 1),     # acted, correct
        _row("t1", 0.80, "wrong", 0, 1),   # acted, incorrect
        _row("t2", 0.70, "", "", 0),       # r_logprob present, no answer
        _row("t3", 0.60, "", "", 0),       # r_logprob present, no answer
    ]
    p = tmp_path / "m" / "m_s1.csv"
    _write_logprob_csv(p, rows)

    coords = logprob_model_coords([str(p)])

    # Answer-parse A = 2 acted / 4 tasks = 0.5
    assert coords["A"] == pytest.approx(0.5)

    # The WRONG (old, drift-back) predicate — "r_logprob non-blank" — would
    # mark all 4 rows acted, giving A = 1.0. This assertion fails loudly if
    # the loader ever regresses to the archived r_selected-blank style.
    n_rlogprob_nonblank = sum(
        1 for r in rows if str(r["r_logprob"]).strip() not in ("", "nan", "none")
    )
    wrong_A = n_rlogprob_nonblank / len(rows)
    assert wrong_A == pytest.approx(1.0)
    assert coords["A"] != pytest.approx(wrong_A)


# ===== 2. Hand-computable H / C / A on a small fixture =====

def test_hand_computable_hca(tmp_path):
    """Exact H, C, A on a 5-task fixture.

    rows (r_logprob, answer, y, acted):
      t0  0.9  "a"  1  1   acted, correct
      t1  0.8  "b"  0  1   acted, incorrect
      t2  0.7  "c"  1  1   acted, correct
      t3  0.5  ""   "" 0   abstained (no answer)
      t4  0.4  ""   "" 0   abstained (no answer)

    A = 3 acted / 5 = 0.6
    H = 2 (acted & y==1) / 5 = 0.4
    Brier over acted = [(0.9-1)^2, (0.8-0)^2, (0.7-1)^2]
                     = [0.01, 0.64, 0.09]  mean = 0.74/3
    C = 1 - 0.74/3 = 0.753333...
    n_acted = 3
    """
    rows = [
        _row("t0", 0.9, "a", 1, 1),
        _row("t1", 0.8, "b", 0, 1),
        _row("t2", 0.7, "c", 1, 1),
        _row("t3", 0.5, "", "", 0),
        _row("t4", 0.4, "", "", 0),
    ]
    p = tmp_path / "m" / "m_s1.csv"
    _write_logprob_csv(p, rows)

    coords = logprob_model_coords([str(p)])
    assert coords["A"] == pytest.approx(0.6)
    assert coords["H"] == pytest.approx(0.4)
    expected_c = 1.0 - (0.01 + 0.64 + 0.09) / 3.0
    assert coords["C"] == pytest.approx(expected_c)
    assert coords["n_acted"] == 3
    assert coords["n_tasks"] == 5


# ===== 3. Single-source: H, C, A all from the SAME rows; no cross-run mix =

def test_single_source_no_cross_model_pooling(tmp_path):
    """A model's (H,C,A) is derived only from ITS own rows. Mixing another
    model's rows would change the values; the loader must not pool across
    models. all_logprob_model_coords keeps models separate."""
    runs = tmp_path / "logprob_xmodel"
    # model A: all acted & correct, r_logprob 1.0 -> A=1, H=1, C=1
    _write_logprob_csv(runs / "modelA" / "modelA_s1.csv", [
        _row("t0", 1.0, "a", 1, 1),
        _row("t1", 1.0, "b", 1, 1),
    ])
    # model B: all abstained -> A=0, H=0, C=nan
    _write_logprob_csv(runs / "modelB" / "modelB_s1.csv", [
        _row("t0", 0.3, "", "", 0),
        _row("t1", 0.2, "", "", 0),
    ])

    coords = all_logprob_model_coords(str(runs))
    assert set(coords) == {"modelA", "modelB"}

    a = coords["modelA"]
    assert a["A"] == pytest.approx(1.0)
    assert a["H"] == pytest.approx(1.0)
    assert a["C"] == pytest.approx(1.0)

    b = coords["modelB"]
    assert b["A"] == pytest.approx(0.0)
    assert b["H"] == pytest.approx(0.0)
    assert math.isnan(b["C"])

    # Had the loader pooled both models, A would be 0.5 (2 acted of 4),
    # which differs from BOTH per-model A's — proves no cross-model mixing.
    assert a["A"] != pytest.approx(0.5)
    assert b["A"] != pytest.approx(0.5)


# ===== 4. Partial-model flag (mirror existing model_points semantics) =====

def test_partial_flag_lt5_seeds(tmp_path):
    """< 5 seed CSVs -> partial=True (mirrors model_points._REQUIRED_SEEDS)."""
    p = tmp_path / "m" / "m_s1.csv"
    _write_logprob_csv(p, [_row("t0", 0.9, "a", 1, 1)])
    coords = logprob_model_coords([str(p)])
    assert coords["partial"] is True
    assert coords["n_seeds"] == 1


def test_partial_flag_short_seed(tmp_path):
    """A seed CSV with < 100 rows -> partial=True even with 5 seeds."""
    paths = []
    for s in range(1, 6):
        p = tmp_path / "m" / f"m_s{s}.csv"
        _write_logprob_csv(p, [_row("t0", 0.9, "a", 1, 1)])  # 1 row << 100
        paths.append(str(p))
    coords = logprob_model_coords(paths)
    assert coords["n_seeds"] == 5
    assert coords["partial"] is True


# ===== 5. Bootstrap CI shape / ordering / nan-safety =====

def test_bootstrap_ci_shape_and_ordering(tmp_path):
    """lo <= point <= hi (roughly) for H and A; C CI is (nan,nan) when no
    acted rows exist anywhere (nan-safe, mirrors _bootstrap_ci)."""
    rows = [_row(f"t{i}", 0.7, "a", (1 if i % 2 == 0 else 0), 1)
            for i in range(40)]
    p = tmp_path / "m" / "m_s1.csv"
    _write_logprob_csv(p, rows)
    coords = logprob_model_coords([str(p)], random_state=0)

    for axis in ("H", "A", "C"):
        lo, hi = coords[f"{axis}_ci"]
        assert lo <= hi + 1e-9
        pt = coords[axis]
        assert lo - 1e-6 <= pt <= hi + 1e-6

    # No acted rows anywhere -> C is nan and C CI is (nan, nan).
    p2 = tmp_path / "z" / "z_s1.csv"
    _write_logprob_csv(p2, [_row("t0", 0.5, "", "", 0)])
    z = logprob_model_coords([str(p2)])
    assert math.isnan(z["C"])
    clo, chi = z["C_ci"]
    assert math.isnan(clo) and math.isnan(chi)


# ===== 6. HONEST_CAPTION_LOGPROB present + required caveat substrings =====

def test_honest_caption_logprob_caveats():
    """Caption-enforced-by-test (mirrors the existing HONEST_CAPTION
    discipline). HONEST_CAPTION_LOGPROB MUST state, for the logprob path:
    descriptive-not-proof; A = answer-commitment via ANSWER-parse and
    conflates deliberate deferral with parse-fail; ungated N=1 so A is
    design-pinned (not a trade-off); (H,C,A) competence-confounded; causal
    claim rests on the gated mechanism experiment + theory; and that C is an
    independent reimplementation of the manuscript logprob-confidence Eq.,
    NOT bit-verified against the original 540-config run.
    The existing HONEST_CAPTION is unchanged (still has its own caveats)."""
    assert isinstance(HONEST_CAPTION_LOGPROB, str)
    assert HONEST_CAPTION_LOGPROB.strip()
    low = HONEST_CAPTION_LOGPROB.lower()

    # descriptive, not a proof
    assert "descriptive" in low
    assert "not" in low
    assert ("proof" in low) or ("impossib" in low) or ("region" in low)

    # answer-commitment via ANSWER-parse
    assert "answer-commitment" in low or "answer commitment" in low
    assert "answer" in low and "parse" in low

    # deferral-vs-parsefail caveat (logprob-reworded)
    assert ("defer" in low) or ("deferral" in low)
    assert (
        ("parse-fail" in low)
        or ("parse fail" in low)
        or ("unparseable" in low)
        or ("parsing failure" in low)
    )

    # ungated N=1 -> design-pinned, not a trade-off
    assert "ungated" in low or "no gat" in low or "no abstention" in low
    assert "n=1" in low or "n = 1" in low
    assert "design-pinned" in low or "design pinned" in low or "not a trade" in low

    # competence-confounded
    assert "competence" in low

    # causal claim rests on the gated mechanism experiment + theory
    assert "mechanism" in low
    assert "theory" in low

    # C is an independent reimplementation, NOT bit-verified vs original
    assert "reimplement" in low or "independent" in low
    assert "bit-verified" in low or "bit verified" in low
    assert "540" in low

    # The existing HONEST_CAPTION is a DIFFERENT, unchanged string.
    assert HONEST_CAPTION_LOGPROB != HONEST_CAPTION
    assert "answer-commitment rate" in HONEST_CAPTION.lower()


# ===== 7. eval_logprob script: mocked generation, exact schema, no net =====

def _fake_completion(content, confidence):
    """Stand-in for ollama_logprob_client.LogprobCompletion."""
    from src.ollama_logprob_client import LogprobCompletion
    return LogprobCompletion(content=content, token_logprobs=[-0.1, -0.2],
                             confidence=confidence)


def test_eval_logprob_writes_exact_schema_answer_parse_acted(tmp_path):
    """With generate_with_logprobs mocked (NO network), running on a 3-task
    fixture writes a CSV with the EXACT 7-column schema and answer-parse
    acted (a confidence-bearing but answer-less completion -> acted=0)."""
    import scripts.eval_logprob as elp

    # 3-task fixture mirroring tasks/task_set.json schema.
    task_set = [
        {"id": "arith_easy_01", "category": "arithmetic_easy",
         "prompt": "p1", "ground_truth": "788", "verification": "arithmetic",
         "expression": "664 + 124"},
        {"id": "fact_common_01", "category": "factual_common",
         "prompt": "p2", "ground_truth": "Au", "verification": "exact"},
        {"id": "fact_common_02", "category": "factual_common",
         "prompt": "p3", "ground_truth": "Tokyo", "verification": "exact"},
    ]
    ts_path = tmp_path / "task_set.json"
    ts_path.write_text(json.dumps(task_set))
    out_dir = tmp_path / "logprob_xmodel"

    # Completions: task0 correct, task1 wrong-answer, task2 NO ANSWER line
    # (confidence is still present — that's the whole point of the pin).
    replies = {
        "p1": _fake_completion("CONFIDENCE: 0.9\nANSWER: 788", 0.91),
        "p2": _fake_completion("CONFIDENCE: 0.8\nANSWER: Berlin", 0.70),
        "p3": _fake_completion("I am not going to answer this.", 0.55),
    }

    def fake_gen(prompt, model=None, temperature=None, seed=None, **kw):
        return replies[prompt]

    with patch.object(elp, "generate_with_logprobs", side_effect=fake_gen):
        out_csv = elp.run_eval(
            model="qwen2.5:7b", seed=7,
            task_set_path=str(ts_path), output_dir=str(out_dir),
        )

    rows = list(csv.DictReader(open(out_csv)))
    assert list(rows[0].keys()) == LOGPROB_HEADER  # EXACT 7-col schema
    assert len(rows) == 3

    by_task = {r["task"]: r for r in rows}

    r0 = by_task["arith_easy_01"]
    assert r0["category"] == "arithmetic_easy"
    assert r0["seed"] == "7"
    assert r0["answer"] == "788"
    assert r0["acted"] == "1"
    assert r0["y"] == "1"
    assert float(r0["r_logprob"]) == pytest.approx(0.91)

    r1 = by_task["fact_common_01"]
    assert r1["answer"] == "Berlin"
    assert r1["acted"] == "1"
    assert r1["y"] == "0"          # wrong answer
    assert float(r1["r_logprob"]) == pytest.approx(0.70)

    # task2: confidence present (0.55) but NO parseable answer -> acted=0.
    r2 = by_task["fact_common_02"]
    assert r2["answer"] == ""
    assert r2["acted"] == "0"
    assert float(r2["r_logprob"]) == pytest.approx(0.55)

    # Path layout: <out>/<model>/<model>_s<seed>.csv
    assert out_csv.endswith("/qwen2.5_7b/qwen2.5_7b_s7.csv") or \
        out_csv.endswith("\\qwen2.5_7b\\qwen2.5_7b_s7.csv")
