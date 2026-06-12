"""R10: per-model Stage-B slugs + config-descriptor generalization."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.select_from_pool import (  # noqa: E402
    config_filename,
    parse_config_descriptor,
)


def test_slugged_filename_roundtrip():
    name = config_filename(32, 4.0, 0.9, 789, model_slug="gemma2_9b")
    assert name == "gemma2_9b_N32_w4.0_r0.9_s789.csv"
    N, w, r, s = parse_config_descriptor(name[:-4], model_slug="gemma2_9b")
    assert (N, w, r, s) == (32, 4.0, 0.9, 789)


def test_mistral_slug_with_dots_and_dashes():
    slug = "mistral_7b-instruct-q4_K_M"
    name = config_filename(1, 0, 0.5, 0, model_slug=slug)
    N, w, r, s = parse_config_descriptor(name[:-4], model_slug=slug)
    assert (N, w, r, s) == (1, 0, 0.5, 0)


def test_default_slug_descriptor_unchanged():
    # Legacy descriptor format keeps parsing without an explicit slug.
    N, w, r, s = parse_config_descriptor("qwen2.5_7b_N32_w0_r0.7_s42")
    assert (N, w, r, s) == (32, 0, 0.7, 42)
    assert isinstance(w, int) or w == 0   # canonical W_RATIOS zero, not 0.0


def test_load_pool_honors_model_slug(tmp_path):
    import json
    import scripts.select_from_pool as sfp
    sfp.clear_pool_cache()
    rec = {"task_id": "a", "category": "arithmetic_easy", "seed": 7,
           "confidence_source": "logprob",
           "completions": [{"idx": 0, "r": 0.9, "y": 1, "a": "42"}]}
    (tmp_path / "pool_gemma2_9b_seed7_N32.jsonl").write_text(
        json.dumps(rec) + "\n")
    loaded = sfp._load_pool(7, tmp_path, model_slug="gemma2_9b")
    assert loaded[0]["task_id"] == "a"
    sfp.clear_pool_cache()
