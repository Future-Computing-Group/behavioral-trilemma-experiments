"""Tests for scripts/generate_pool.py — Stage-A pool reconstruction.

Stage A generates one pool record per (task, seed) by calling Ollama
``generate_with_logprobs`` ``N_MAX`` times with per-completion seeds
``base_seed + i``, parses verbalized + logprob confidence and the answer,
verifies y against ground truth, and writes one jsonl record per task to
``pool_qwen2.5_7b_seed{S}_N32.jsonl``.

These tests are network-free: they mock ``generate_with_logprobs`` and
``_verify_answer``. The real-Ollama smoke checks live in the driver script's
``--smoke`` flag, not in pytest, so the unit suite stays fast and runnable
in CI.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_pool import (  # noqa: E402
    build_completion_record,
    build_pool_record,
    POOL_COMPLETION_FIELDS,
    POOL_RECORD_FIELDS,
)


# ---------------------------------------------------------------------------
# Mocks: a fake LogprobCompletion + a fake generator + a fake verifier
# ---------------------------------------------------------------------------

class FakeLogprobCompletion:
    """Shape-compatible with src.ollama_logprob_client.LogprobCompletion."""
    def __init__(self, content, token_logprobs):
        self.content = content
        self.token_logprobs = token_logprobs
        # exp(mean), clipped to [0.01, 1.0]: match the production confidence.
        mean = sum(token_logprobs) / len(token_logprobs)
        self.confidence = min(1.0, max(0.01, math.exp(mean)))


# ---------------------------------------------------------------------------
# build_completion_record: per-completion record schema + field semantics
# ---------------------------------------------------------------------------

def test_completion_record_schema_keys_in_exact_order():
    comp = FakeLogprobCompletion(
        content="CONFIDENCE: 0.85\nANSWER: 42",
        token_logprobs=[-0.05, -0.10, -0.05],
    )
    rec = build_completion_record(idx=0, completion=comp, y=1)
    # Pool jsonl schema (verified against pool_qwen2.5_7b_seed0_N32.jsonl
    # record code_simple_11 completion[0] etc.):
    #   {idx, raw, r, r_verbalized, r_logprob, mean_logprob, a, y}
    assert list(rec.keys()) == POOL_COMPLETION_FIELDS
    assert len(POOL_COMPLETION_FIELDS) == 8


def test_completion_record_parses_verbalized_and_answer():
    comp = FakeLogprobCompletion(
        content="CONFIDENCE: 0.85\nANSWER: 42",
        token_logprobs=[-0.05, -0.10, -0.05],
    )
    rec = build_completion_record(idx=7, completion=comp, y=1)
    assert rec["idx"] == 7
    assert rec["raw"] == "CONFIDENCE: 0.85\nANSWER: 42"
    assert rec["a"] == "42"
    assert rec["r_verbalized"] == 0.85
    # r = r_logprob = exp(mean(token_logprobs)) clipped
    assert rec["r"] == pytest.approx(math.exp(-0.0666666), rel=1e-3)
    assert rec["r_logprob"] == rec["r"]
    assert rec["mean_logprob"] == pytest.approx(-0.0666666, rel=1e-3)
    assert rec["y"] == 1


def test_completion_record_r_equals_r_logprob_canonically():
    # The pool's ``confidence_source == "logprob"`` => r is r_logprob, NOT
    # r_verbalized. Verified against the real pool record where
    # r=0.999983 == r_logprob=0.999983 != r_verbalized=1.0.
    comp = FakeLogprobCompletion(
        content="CONFIDENCE: 1.0\nANSWER: 42",
        token_logprobs=[-0.00001] * 10,
    )
    rec = build_completion_record(idx=0, completion=comp, y=1)
    assert rec["r"] == rec["r_logprob"]
    assert rec["r_verbalized"] == 1.0
    assert rec["r"] != rec["r_verbalized"]  # they differ in the canonical case


def test_completion_record_a_none_serialised_as_json_null():
    # When the ANSWER:-parser returns None (no parseable answer), the pool
    # record stores a=None (which json serialises as null). Real example:
    # code_simple_11 seed=456 idx=21 had a=None in the shipped pool.
    comp = FakeLogprobCompletion(
        content="CONFIDIENCE: 1.0\nANWSER: foo",  # typo => parser misses
        token_logprobs=[-0.05] * 5,
    )
    rec = build_completion_record(idx=21, completion=comp, y=0)
    assert rec["a"] is None
    assert rec["r_verbalized"] is None  # CONFIDENCE: line absent => also None


def test_completion_record_y_is_passed_through_not_computed_internally():
    # Decoupling: build_completion_record does NOT call the verifier; it
    # receives y from the caller. Verifier wiring is tested separately so
    # this function stays pure and trivially mockable.
    comp = FakeLogprobCompletion(
        content="CONFIDENCE: 0.5\nANSWER: WRONG",
        token_logprobs=[-1.0, -1.0],
    )
    rec = build_completion_record(idx=0, completion=comp, y=99)  # nonsensical y
    assert rec["y"] == 99  # passed straight through


# ---------------------------------------------------------------------------
# build_pool_record: per-task assembly across N completions
# ---------------------------------------------------------------------------

def test_pool_record_schema_keys_in_exact_order():
    task = {"id": "t1", "category": "arithmetic_easy", "prompt": "...",
            "ground_truth": "42", "verification": "exact"}
    comps = [FakeLogprobCompletion("CONFIDENCE: 0.9\nANSWER: 42", [-0.1])]
    rec = build_pool_record(
        task=task, seed=0, completions=comps,
        verify_y=lambda t, a: 1 if a == "42" else 0,
    )
    # Pool jsonl record shape: {task_id, category, seed, confidence_source,
    # completions}
    assert list(rec.keys()) == POOL_RECORD_FIELDS


def test_pool_record_assembles_all_N_completions_with_correct_idx():
    task = {"id": "t1", "category": "arithmetic_easy", "prompt": "...",
            "ground_truth": "42", "verification": "exact"}
    comps = [
        FakeLogprobCompletion(f"CONFIDENCE: 0.9\nANSWER: {a}", [-0.1])
        for a in ["42", "41", "42", "WRONG"]
    ]
    rec = build_pool_record(
        task=task, seed=42, completions=comps,
        verify_y=lambda t, a: 1 if a == t["ground_truth"] else 0,
    )
    assert rec["task_id"] == "t1"
    assert rec["category"] == "arithmetic_easy"
    assert rec["seed"] == 42
    assert rec["confidence_source"] == "logprob"
    assert len(rec["completions"]) == 4
    assert [c["idx"] for c in rec["completions"]] == [0, 1, 2, 3]
    assert [c["a"] for c in rec["completions"]] == ["42", "41", "42", "WRONG"]
    assert [c["y"] for c in rec["completions"]] == [1, 0, 1, 0]


def test_pool_record_verifier_is_called_per_completion():
    # The verifier is called once per completion with (task, parsed_answer).
    task = {"id": "t1", "category": "arithmetic_easy", "prompt": "...",
            "ground_truth": "42", "verification": "exact"}
    comps = [
        FakeLogprobCompletion("CONFIDENCE: 0.9\nANSWER: 42", [-0.1]),
        FakeLogprobCompletion("CONFIDENCE: 0.9\nANSWER: 41", [-0.1]),
    ]
    calls = []
    def verify(t, a):
        calls.append((t["id"], a))
        return 1 if a == "42" else 0
    build_pool_record(task=task, seed=0, completions=comps, verify_y=verify)
    assert calls == [("t1", "42"), ("t1", "41")]


# ---------------------------------------------------------------------------
# JSON round-trip: the record must survive jsonl serialisation losslessly
# ---------------------------------------------------------------------------

def test_record_jsonl_round_trips_losslessly():
    task = {"id": "t1", "category": "arithmetic_easy", "prompt": "...",
            "ground_truth": "42", "verification": "exact"}
    comps = [
        FakeLogprobCompletion("CONFIDENCE: 0.9\nANSWER: 42",
                              [-0.001, -0.002, -0.003]),
    ]
    rec = build_pool_record(
        task=task, seed=0, completions=comps,
        verify_y=lambda t, a: 1,
    )
    line = json.dumps(rec)
    back = json.loads(line)
    assert back == rec
    # Confirm None survives as null in jsonl
    comps_with_none = [FakeLogprobCompletion("garbage", [-0.5])]
    rec2 = build_pool_record(
        task=task, seed=0, completions=comps_with_none,
        verify_y=lambda t, a: 0,
    )
    back2 = json.loads(json.dumps(rec2))
    assert back2["completions"][0]["a"] is None
    assert back2["completions"][0]["r_verbalized"] is None


# ---------------------------------------------------------------------------
# Reference-pool replication: regenerate one completion record from the
# real shipped pool to confirm field-by-field semantic match. (No network:
# we use the raw text + token logprobs that ARE preserved in the pool, so
# we can re-run the parsing/scoring locally without calling Ollama.)
# ---------------------------------------------------------------------------

REAL_POOL = (ROOT / "experiment_output" / "raw_runs" / "logprob" / "pools"
             / "pool_qwen2.5_7b_seed0_N32.jsonl")


@pytest.mark.skipif(not REAL_POOL.exists(),
                    reason="real pool jsonl not present (gitignored)")
def test_replays_shipped_pool_completion_field_by_field():
    """Read a real pool record's first completion; build a fake completion
    with that completion's content + token logprobs (NOT preserved in pool
    jsonl, so we synthesise from mean_logprob to match the same r); confirm
    the build_completion_record output matches the shipped fields modulo
    token_logprobs (which we can't recover; mean_logprob is preserved).

    The pool jsonl stores only mean_logprob, not the full token-logprobs
    list, so true byte-equality requires re-running Ollama. This test
    instead confirms: given the SAME raw text and the SAME mean_logprob,
    our builder produces the same r, r_logprob, r_verbalized, a, y as
    the shipped record."""
    with REAL_POOL.open() as f:
        first_record = json.loads(f.readline())
    shipped = first_record["completions"][0]

    # Construct token_logprobs that average to the shipped mean_logprob.
    n_tokens = 10
    token_logprobs = [shipped["mean_logprob"]] * n_tokens
    fake = FakeLogprobCompletion(content=shipped["raw"], token_logprobs=token_logprobs)
    rec = build_completion_record(idx=shipped["idx"], completion=fake, y=shipped["y"])

    assert rec["idx"] == shipped["idx"]
    assert rec["raw"] == shipped["raw"]
    assert rec["a"] == shipped["a"]
    assert rec["r_verbalized"] == shipped["r_verbalized"]
    # r and r_logprob: clip-and-exp on the same mean must reproduce the value
    assert rec["r_logprob"] == pytest.approx(shipped["r_logprob"], rel=1e-6)
    assert rec["r"] == pytest.approx(shipped["r"], rel=1e-6)
    assert rec["mean_logprob"] == pytest.approx(shipped["mean_logprob"], rel=1e-6)
    assert rec["y"] == shipped["y"]
