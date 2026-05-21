# Behavioral Credibility Trilemma: Empirical Validation

Experiment code and raw results for the empirical validation of the Behavioral
Credibility Trilemma via Best-of-N selection. This repository accompanies the
manuscript

> L. Lovén, N. Do, H. Mehmood, S. Tarkoma (2026). *The Behavioral
> Credibility Trilemma: When Calibrated Autonomy Becomes Impossible.*
> Journal of Machine Learning Research (under review).

## What's here

A 540-configuration Best-of-N sweep on Qwen-2.5-7B (54,000 selected-task
observations) testing five pre-registered hypotheses derived from the
Behavioral Perturbation Lemma, plus a descriptive analysis of the
achievable-(H, C, A) surface geometry of the Confidence-Gated Decision
Problem. All five hypotheses are confirmed at α = 0.05 after
Bonferroni–Holm correction. The repository ships the full simulation code,
the task set, the experiment configs, the hypothesis-results JSON (the
Table-1 values), and the H3 figure. The raw per-completion CSVs are not a
deterministic function of the code (the run is a stochastic LLM-in-the-loop
process at temperature τ = 0.8), so they are regenerable via the pipeline
(Stages 1–2) rather than shipped; re-runs reproduce the effects, not the
exact records.

## Reproducing the paper

The full pipeline consists of three stages:

### Stage 1: Phase 0 calibration (held-out)

Estimates per-task base accuracy $\hat{p}_t$ and binding-set membership at
each threshold $r_{\min} \in \{0.5, 0.7, 0.9\}$. Uses 20 held-out seeds
`{1000..1019}`, disjoint from the experimental seeds to avoid circularity in
the H5 binding-state-specificity test.

```bash
python -m scripts.run --mode phase0
# Output: experiment_output/raw_runs/logprob/results/phase0_calibration.csv
```

Estimated runtime: ~1 hour on a single machine.

### Stage 2: Full 540-config Best-of-N sweep

```bash
python -m scripts.run --mode full
# Output: 540 per-configuration CSVs under
#         experiment_output/raw_runs/logprob/results/
```

Sweep: $N \in \{1, 2, 4, 8, 16, 32\}$, $w_A/w_C \in \{0, 0.25, 0.5, 1, 2, 4\}$,
$r_{\min} \in \{0.5, 0.7, 0.9\}$, seeds $\{0, 42, 123, 456, 789\}$.
Total configurations: $6 \times 6 \times 3 \times 5 = 540$.

Estimated runtime: ~14 hours on a single machine with Ollama serving
Qwen-2.5-7B locally.

### Stage 3: Analysis and figures

```bash
# Rebuild hypothesis_results.json end-to-end from raw CSVs
python -m scripts.regenerate_hypothesis_results
# Output: experiment_output/analysis/hypothesis_results.json (rewritten in place)

# Plot H3 achievable-region convexity violation rate by N (descriptive)
python -m scripts.plot_h3_convexity_by_N
# Output: experiment_output/analysis/figures/h3_convexity_by_N.{pdf,png}
```

`N_BOOT` environment variable controls bootstrap-CI resamples for H1/H5/H6
(default 2000; the paper uses 10000 in a final run but 2000 is within
measurement noise and much faster):

```bash
N_BOOT=10000 python -m scripts.regenerate_hypothesis_results
```

Expected runtime: 1–3 minutes for `regenerate_hypothesis_results.py` at
`N_BOOT=2000`, ~5 minutes at `N_BOOT=10000`. Plot script: seconds.

After Stage 3, the `hypothesis_results.json` keys should match Table 1 of
the manuscript. H1, H2, H4, H5, H6 are the **five pre-registered hypothesis
tests** (all confirmed); **H3 is reported as the descriptive
surface-geometry analysis** of the achievable-(H, C, A) region, not a
confirmed test:

| Hypothesis | $p$-value | Effect size |
|---|---|---|
| H1 Fixed-axis gating degradation | $4.67 \times 10^{-19}$ | $d = 1.10$ |
| H2 Monotone inflation trend (Jonckheere–Terpstra) | $8.49 \times 10^{-5}$ | $\rho = 0.89$ |
| H3 Achievable-region convexity (descriptive) | binomial test, 10% < 15% | — |
| H4 Threshold clustering | $< 10^{-3}$ | $z = 30.02$ |
| H5 Binding-state specificity | $< 10^{-3}$ | $d = 5.32$ |
| H6 Control ($w_A = 0$) | $1.35 \times 10^{-23}$ | $d = 1.31$ |

## Repository structure

```
LICENSE
README.md
requirements.txt
EXPERIMENT-PLAN.md                           # full protocol (§4.3 has Phase 0 details)
analysis/
  hypothesis_tests.py                        # H1–H6 tests + helpers
  metrics.py                                 # Brier decomposition etc.
  figures.py                                 # general figure utilities
configs/
  params.yaml                                # weight grid, seeds, r_min
scripts/
  run.py                                     # orchestrator (phase0 + full)
  regenerate_hypothesis_results.py           # rebuild JSON from raw CSVs
  plot_h3_convexity_by_N.py                  # H3 stratified-by-N figure
  generate_tasks.py                          # task generation
src/
  orchestrator.py                            # per-config runner
  ollama_client.py                           # Ollama OpenAI-compatible client
  parser.py                                  # response parser
  scorer.py                                  # composite payoff
tasks/                                       # 100 tasks (arith/factual/code)
tests/                                       # pytest unit tests
experiment_output/
  analysis/                                  # canonical results (paper's Table 1)
    hypothesis_results.json                  #   shipped; rewritten by Stage 3
    figures/h3_convexity_by_N.{pdf,png}      #   shipped; manuscript Figure 2
    (aggregate metric CSVs regenerated here by Stage 3)
  competence_probe/figures/model_points.*    # shipped; model-points figure
  logprob_xmodel/<model>/<model>_s{0..4}.csv # shipped; cross-model logprob CSVs
  raw_runs/logprob/results/                  # NOT shipped — regenerated by Stages 1-2
                                             #   (540 per-config CSVs + phase0)
docs/
  REPRODUCING.md                             # step-by-step reproduction guide
```

## Dependencies

- Python 3.10 or newer (tested on 3.11)
- Ollama 0.4.2 or newer, with `qwen2.5:7b` pulled (`ollama pull qwen2.5:7b`)
  - The paper uses the default Ollama quantization, Q4_K_M
  - Inference via the OpenAI-compatible endpoint (`/v1/chat/completions`)
    with `logprobs: true`; temperature $\tau = 0.8$
- Python packages: `pip install -r requirements.txt`

## Confidence metric

The per-completion confidence report $r_i$ is derived from token-level
log-probabilities returned by Ollama:

$$r_i = \exp\!\left(\frac{1}{T}\sum_{t=1}^{T} \ell_t\right) = \left(\prod_{t=1}^{T} p_t\right)^{1/T}$$

the geometric mean of per-token probabilities, clipped to $[0.01, 1.0]$.
See the manuscript §7 and [`src/parser.py`](src/parser.py) for the exact
extraction code.

## Ground-truth verification

The oracle correctness label $y_i \in \{0, 1\}$ is set task-type-specifically:

- **Arithmetic:** exact-value comparison after numeric parsing
- **Factual:** matched against the curated reference file
  `tasks/factual_truth.csv`
- **Code:** Python test-case execution

See [`src/scorer.py`](src/scorer.py) for the full verification logic.

## Legacy: `experiment_output/raw_runs/qwen_2.5/`

The `qwen_2.5/` subdirectory contains results from an earlier run using
verbalized confidence (the model states its own confidence in prose). The
`logprob/` results supersede these. The legacy directory is kept for
audit-trail purposes and should not be used for the paper's analysis
pipeline; all `regenerate_hypothesis_results.py` and `plot_*` scripts load
exclusively from `logprob/`.

## Citation

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

## License

MIT — see [LICENSE](LICENSE).

## Contact

Issues and questions: please open a GitHub issue.
