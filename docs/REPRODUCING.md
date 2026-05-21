# Reproducing the paper

Step-by-step guide to reproducing the 540-configuration Best-of-N sweep and the
analysis (five pre-registered hypothesis tests, plus the descriptive H3
surface-geometry analysis) reported in the JMLR A1 manuscript
*The Behavioral Credibility Trilemma: When Calibrated Autonomy Becomes
Impossible* (Lovén, Do, Mehmood, and Tarkoma, 2026).

The raw per-completion CSVs are not shipped (the run is a stochastic
LLM-in-the-loop process; re-runs reproduce the effects, not the exact
records). To rebuild from your own run, do Stages 1–2 first; the
hypothesis-results JSON shipped in the repository holds the paper's
Table-1 values.

---

## 0. Prerequisites

- **Python** 3.10 or newer (tested on 3.11 on macOS 12+; expected to work on
  Linux and Windows under equivalent Python versions).
- **Ollama** 0.4.2 or newer. Install from <https://ollama.com>.
- **Git** and ~6 GB of disk (model weights; raw CSVs are regenerated locally).
- A machine with ~64 GB RAM if you intend to run the full 540-config sweep
  locally; 16 GB suffices for the analysis-only reproduction.

## 1. Clone the repository and install Python dependencies

```bash
git clone https://github.com/lloven/behavioral-trilemma-experiments.git
cd behavioral-trilemma-experiments

python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

## 2. Prepare Ollama and the Qwen-2.5-7B model

Start the Ollama server (leave it running in a second terminal):

```bash
ollama serve
```

Pull the model (default Q4_K_M quantization — the one the paper uses):

```bash
ollama pull qwen2.5:7b
```

Verify:

```bash
ollama list
# qwen2.5:7b    ...    4.7 GB    ...
```

## 3. Sanity-check: unit + integration smoke

Fast sanity tests to confirm the pipeline is wired correctly before any long
runs:

```bash
# Unit smoke (seconds): task generators, scorer, parser, metrics
pytest -q

# Integration smoke (~2 minutes): 1 task * 2 seeds * 2 configurations,
# hitting the real Ollama endpoint
python -m scripts.run --mode integration_smoke
```

If unit tests fail, fix those before proceeding. If integration smoke fails,
check that Ollama is running and `qwen2.5:7b` is pulled.

## 4. Stage 1 — Phase 0 calibration (held-out seeds)

Purpose: estimate per-task base accuracy $\hat{p}_t$ and binding-set
membership $\{t : \hat{p}_t < r_{\min}\}$ at each $r_{\min} \in \{0.5, 0.7, 0.9\}$.
Uses 20 held-out seeds `{1000..1019}`, disjoint from the experimental seeds,
to avoid circularity in the H5 binding-state-specificity test.

```bash
python -m scripts.run --mode phase0
```

Estimated runtime: **~1 hour** on a single machine (20 seeds × 100 tasks,
N=1 / w_A=0).

Output:

```
experiment_output/raw_runs/logprob/results/phase0_calibration.csv
```

Columns: `task_id, p_hat, n_runs, binding_0.5, binding_0.7, binding_0.9`.

## 5. Stage 2 — Full 540-configuration Best-of-N sweep

Sweep:

| Axis | Values | Count |
|---|---|---|
| $N$ (selection size) | 1, 2, 4, 8, 16, 32 | 6 |
| $w_A/w_C$ (weight ratio) | 0, 0.25, 0.5, 1, 2, 4 | 6 |
| $r_{\min}$ (gate threshold) | 0.5, 0.7, 0.9 | 3 |
| seed | 0, 42, 123, 456, 789 | 5 |
| **Total** |  | **540** |

```bash
python -m scripts.run --mode full
```

Estimated runtime: **~14 hours** on a single machine with Ollama serving
Qwen-2.5-7B locally (temperature $\tau = 0.8$, `logprobs: true`, OpenAI-
compatible endpoint).

Output: 540 per-configuration CSVs under `experiment_output/raw_runs/logprob/results/`,
one per `(N, w_ratio, r_min, seed)` cell, plus aggregated metrics written by
the driver at the end:

```
experiment_output/
├── analysis/
│   ├── brier_decomposition.csv
│   ├── inflation_metrics.csv
│   ├── pairwise_effect_sizes.csv
│   └── per_config_metrics.csv
└── raw_runs/logprob/results/
    ├── phase0_calibration.csv
    └── qwen2.5_7b_N{1,2,4,8,16,32}_w{0,0.25,...}_r{0.5,0.7,0.9}_s{0,42,...}.csv
```

The raw 540 CSVs are not shipped (stochastic LLM run; not bit-reproducible).
Stage 2 regenerates them locally; the paper's hypothesis-results JSON is
shipped for comparison.

## 6. Stage 3 — Analysis and figures

Regenerate `hypothesis_results.json` end-to-end from the raw CSVs:

```bash
python -m scripts.regenerate_hypothesis_results
# default N_BOOT=2000

# For tighter bootstrap CIs (slower):
N_BOOT=10000 python -m scripts.regenerate_hypothesis_results
```

Runtime: **1–3 minutes** at `N_BOOT=2000`, **~5 minutes** at `N_BOOT=10000`.
Output:

```
experiment_output/analysis/hypothesis_results.json
```
(rewritten in place — this is the file the paper's Table 1 is drawn from)

Generate the H3 stratified-by-$N$ figure:

```bash
python -m scripts.plot_h3_convexity_by_N
```

Runtime: seconds. Output:

```
experiment_output/analysis/figures/
├── h3_convexity_by_N.pdf
└── h3_convexity_by_N.png
```

## 7. Verify the results match the manuscript

The JSON's top-level keys should match Table 1 of the manuscript. H1, H2,
H4, H5, H6 are the five pre-registered hypothesis tests; H3 is the
descriptive surface-geometry analysis (reported, not a confirmed test):

| Key | Expected | Manuscript |
|---|---|---|
| `H1.p_value` | ~4.67e-19 | Table 1 H1 |
| `H1.effect_size` | ~1.10 | Table 1 H1 |
| `H2.p_value` | ~8.5e-5 | Table 1 H2 |
| `H2.rho` | ~0.89 | Table 1 H2 |
| `H3.violation_rate` | 0.10 | Table 1 H3 |
| `H3.criterion_met` | `true` (10% < 15%) | Table 1 H3 |
| `H4[0.7].z` | ~30.02 | Table 1 H4 |
| `H5.effect_size` | ~5.32 | Table 1 H5 |
| `H6.effect_size` | ~1.31 | Table 1 H6 |

Effect sizes vary slightly with `N_BOOT`; p-values do not.

The H3 figure should match manuscript Figure 2: a monotone-increasing
violation-rate trend from 0% at $N \in \{1, 2\}$ up to 28.3% at $N = 32$,
with the 15% tolerance line crossed between $N = 8$ and $N = 16$.

## 8. Troubleshooting

- **"connection refused"** — Ollama is not running; start `ollama serve`.
- **Unexpected hypothesis numbers** — the canonical results live in
  `experiment_output/analysis/hypothesis_results.json` (the file the paper's
  Table 1 is drawn from). Re-running `regenerate_hypothesis_results` rewrites
  it in place and reproduces Table 1 exactly.
- **Out-of-memory during Stage 2** — reduce `N_max` in
  `configs/params.yaml` and re-run. The 540-config count assumes the full
  grid.
- **`hypothesis_tests.py` changed**: if you edit the analysis code, rerun
  `python -m scripts.regenerate_hypothesis_results` to refresh the JSON.
  The plot script reads the JSON only.

## 9. Citation

```bibtex
@article{loven2026trilemma,
  title   = {The Behavioral Credibility Trilemma: When Calibrated Autonomy
             Becomes Impossible},
  author  = {Lov{\'e}n, Lauri and Do, Nam and Mehmood, Hassan and
             Tarkoma, Sasu},
  journal = {Journal of Machine Learning Research},
  year    = {2026},
  note    = {Under review}
}
```
