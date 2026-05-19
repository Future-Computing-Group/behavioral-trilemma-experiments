"""Tests for config loader. RED phase: these must fail before implementation."""
from src.config import load_config, generate_configs


def test_load_config_returns_dict(params_path):
    cfg = load_config(params_path)
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "experiment" in cfg


def test_load_config_model_fields(params_path):
    cfg = load_config(params_path)
    assert cfg["model"]["primary"] == "qwen2.5:7b"
    assert cfg["model"]["secondary"] == "llama3.1:8b"
    assert cfg["model"]["temperature"] == 0.8


def test_generate_configs_full(params_path):
    cfg = load_config(params_path)
    configs = list(generate_configs(cfg, mode="full"))
    # 6 N × 6 w × 3 r_min × 5 seeds = 540
    assert len(configs) == 540


def test_generate_configs_unit_smoke(params_path):
    cfg = load_config(params_path)
    configs = list(generate_configs(cfg, mode="unit_smoke"))
    # 2 N × 2 w × 1 r_min × 1 seed = 4
    assert len(configs) == 4


def test_generate_configs_integration_smoke(params_path):
    cfg = load_config(params_path)
    configs = list(generate_configs(cfg, mode="integration_smoke"))
    # 3 N × 3 w × 1 r_min × 2 seeds = 18
    assert len(configs) == 18


def test_config_fields_present(params_path):
    cfg = load_config(params_path)
    configs = list(generate_configs(cfg, mode="unit_smoke"))
    c = configs[0]
    assert "N" in c
    assert "w_ratio" in c
    assert "r_min" in c
    assert "seed" in c
    assert "w_C" in c
    assert "w_A" in c


def test_config_w_A_computed_correctly(params_path):
    cfg = load_config(params_path)
    configs = list(generate_configs(cfg, mode="unit_smoke"))
    for c in configs:
        assert c["w_A"] == c["w_C"] * c["w_ratio"]


def test_calibration_seeds_separate(params_path):
    cfg = load_config(params_path)
    exp_seeds = set(cfg["experiment"]["seeds"])
    cal_seeds = set(cfg["calibration"]["seeds"])
    assert exp_seeds.isdisjoint(cal_seeds), "Calibration seeds must not overlap experimental seeds"


def test_competence_probe_seeds_separate(params_path):
    cfg = load_config(params_path)
    exp_seeds = set(cfg["experiment"]["seeds"])
    cal_seeds = set(cfg["calibration"]["seeds"])
    probe_seeds = set(cfg["competence_probe"]["seeds"])
    assert probe_seeds.isdisjoint(exp_seeds), "Competence-probe seeds must not overlap experimental seeds"
    assert probe_seeds.isdisjoint(cal_seeds), "Competence-probe seeds must not overlap calibration seeds"
    for key in ("mistral7b", "gemma2_9b", "qwen14b"):
        assert key in cfg["model"], f"Model registry must include {key}"
