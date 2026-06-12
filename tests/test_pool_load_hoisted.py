"""R8 (L97): Stage-B pool loads hoisted — one read per pool file per sweep.

Counting-loader seam: ``_read_pool_file`` is the uncached inner reader;
``_load_pool`` memoizes on top of it; ``clear_pool_cache()`` resets.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.select_from_pool as sfp  # noqa: E402


def _write_minimal_pool(pools_dir, seed):
    recs = [
        {"task_id": "a", "category": "arithmetic_easy", "seed": seed,
         "confidence_source": "logprob",
         "completions": [{"idx": 0, "r": 0.9, "y": 1, "a": "42"},
                         {"idx": 1, "r": 0.4, "y": 0, "a": "7"}]},
        {"task_id": "b", "category": "factual_common", "seed": seed,
         "confidence_source": "logprob",
         "completions": [{"idx": 0, "r": 0.6, "y": 0, "a": "x"},
                         {"idx": 1, "r": 0.8, "y": 1, "a": "42"}]},
    ]
    path = pools_dir / f"pool_qwen2.5_7b_seed{seed}_N32.jsonl"
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return path


def test_pool_loaded_once_per_seed(monkeypatch, tmp_path):
    sfp.clear_pool_cache()
    loads = []
    real = sfp._read_pool_file          # the new uncached inner reader
    monkeypatch.setattr(sfp, "_read_pool_file",
                        lambda *a, **k: (loads.append(a), real(*a, **k))[1])
    _write_minimal_pool(tmp_path, seed=42)
    for N in (1, 2):
        for w in (0, 4.0):
            sfp.write_config_csv(N, w, 0.7, 42, tmp_path / "out",
                                 pools_dir=tmp_path)
    assert len(loads) == 1
    sfp.clear_pool_cache()


def test_clear_pool_cache_forces_reread(monkeypatch, tmp_path):
    sfp.clear_pool_cache()
    loads = []
    real = sfp._read_pool_file
    monkeypatch.setattr(sfp, "_read_pool_file",
                        lambda *a, **k: (loads.append(a), real(*a, **k))[1])
    _write_minimal_pool(tmp_path, seed=42)
    sfp.write_config_csv(1, 0, 0.7, 42, tmp_path / "out", pools_dir=tmp_path)
    sfp.clear_pool_cache()
    sfp.write_config_csv(2, 0, 0.7, 42, tmp_path / "out", pools_dir=tmp_path)
    assert len(loads) == 2
    sfp.clear_pool_cache()
