"""Tests for the R3 parameterization of regenerate_hypothesis_results.

The script must honor --results-dir / --phase0 / --out / --glob so the
E-A and E-B re-analyses write SIBLING artifacts; the zero-arg invocation
must keep resolving the canonical (primary) paths byte-stably.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_regenerate_honors_out_and_results_dir(tmp_path):
    # Fixture: 4 tiny synthetic per-config CSVs (19-col schema, 3 tasks,
    # N in {1,32}, w in {0,4.0}, r_min=0.7, seed=42) + a 3-task phase0 CSV.
    fix = pathlib.Path(__file__).parent / "fixtures" / "regen_mini"
    res = tmp_path / "results"
    shutil.copytree(fix / "results", res)
    out = tmp_path / "hyp.json"
    cmd = [sys.executable, "-m", "scripts.regenerate_hypothesis_results",
           "--results-dir", str(res), "--phase0", str(fix / "phase0.csv"),
           "--out", str(out), "--glob", "mini_*.csv"]
    env = dict(os.environ, N_BOOT="50")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=ROOT)
    assert out.exists(), r.stderr
    data = json.loads(out.read_text())
    assert "correction" in data and set("H1 H2 H4 H5 H6".split()) <= set(data)


def test_zero_arg_defaults_are_canonical_paths():
    from scripts.regenerate_hypothesis_results import (
        build_parser, RAW_DIR, PHASE0_PATH, OUT_PATH,
    )
    args = build_parser().parse_args([])
    assert args.results_dir == RAW_DIR
    assert args.phase0 == PHASE0_PATH
    assert args.out == OUT_PATH
    assert args.glob == "qwen2.5_7b_*.csv"
