# Behavioral Credibility Trilemma: Empirical Validation

Experiment code for the Best-of-N trilemma manifestation experiment (JMLR Paper A1).

## Overview

Tests three predictions of the Behavioral Credibility Trilemma using Best-of-N
selection as a formal optimization mechanism:

1. **FKG prediction:** Calibration degrades monotonically with selection pressure N
2. **Perturbation Lemma:** Confidence inflation scales with the autonomy/calibration weight ratio
3. **Pareto bound:** Achievable (H,C,A) triples form a convex surface

## Quick start

```bash
# Unit smoke (seconds)
python -m scripts.run_experiment --mode unit_smoke

# Integration smoke (minutes)
python -m scripts.run_experiment --mode integration_smoke

# Full experiment (~14 hours)
python -m scripts.run_experiment --mode full
```

## Structure

```
configs/          # Experiment configurations (params.yaml, weight vectors)
scripts/          # Orchestrator + task runner + scoring
tasks/            # Task sets with ground truth (math, factual, code)
results/          # CSV outputs per configuration
analysis/         # Post-hoc analysis scripts + figures
tests/            # Unit tests for scoring, metrics, task verification
notes/            # Experiment design notes and decisions
```

## Models

- **Primary:** Qwen-2.5-7B via Ollama (local, free, reproducible)
- **Secondary:** Llama-3-8B or Mistral-7B for cross-model validation

## Dependencies

- Python 3.10+
- `requests` (for Ollama API)
- `numpy`, `scipy` (for statistics)
- `pandas` (for results processing)
- Ollama running locally with qwen2.5:7b loaded
