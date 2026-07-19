# Behavioral Credibility Trilemma: Empirical Validation

Experiment code and raw results for the empirical validation of the Behavioral
Credibility Trilemma via Best-of-N selection. This repository accompanies the
manuscript

> L. Lovén, N. Do, H. Mehmood, D. K. Sah, S. Tarkoma (2026). _The Behavioral
> Credibility Trilemma: When Calibrated Autonomy Becomes Impossible._
> Preprint, [arXiv:2605.25739](https://arxiv.org/abs/2605.25739).
> In preparation for submission to the Journal of Machine Learning Research.

## What's here

A 540-configuration Best-of-N sweep on Qwen-2.5-7B (54,000 selected-task
observations) testing five hypotheses derived from the
Behavioral Perturbation Lemma, plus a descriptive analysis of the
achievable-(H, C, A) surface geometry of the Confidence-Gated Decision
Problem. All five hypotheses are confirmed at α = 0.05 after
Bonferroni–Holm correction. The repository ships the full simulation code,
the task set, the experiment configs, the hypothesis-results JSONs (the
Table-1 values plus the robustness/replication siblings), the three
Phase-0 calibration CSVs (qwen / gemma2 / yi binding sets), and the H3
figure. The raw per-completion CSVs are not a
deterministic function of the code (the run is a stochastic LLM-in-the-loop
process at temperature τ = 0.8), so they are regenerable via the pipeline
(Stages 1–2) rather than shipped; re-runs reproduce the effects, not the
exact records.

## Reproducing the paper

The pipeline is **two-stage**: a Stage-A pool generator that calls Ollama
once per `(task, seed, draw)` to capture per-completion text + per-token
logprobs + the parsed answer + oracle correctness; and a Stage-B selector
that runs Best-of-N argmax over the pool for each of the 540 weight-config
cells. Stage A is the expensive step (~14 h of local Ollama); Stage B is
purely deterministic and runs in seconds against the pool jsonls. Plus a
Phase-0 calibration step that estimates per-task base accuracy from a
disjoint held-out seed split.

### Stage 0: Phase-0 calibration (held-out, 20 seeds)

Estimates per-task base accuracy $\hat{p}_t$ and binding-set membership at
each threshold $r_{\min} \in \{0.5, 0.7, 0.9\}$ using 20 held-out seeds
`{1000..1019}`, disjoint from the experimental seeds. Avoids circularity in
the H5 binding-state-specificity test.

```bash
python -m scripts.run --phase0
# Output: experiment_output/raw_runs/qwen_2.5/phase0_calibration.csv
# (the canonical Phase-0 artifact; this exact file is what
#  scripts/regenerate_hypothesis_results.py reads)
```

Estimated runtime: ~1 hour on a single machine.

### Stage A: pool generation (32 completions × 100 tasks × 5 seeds)

Generates the per-completion pool jsonl files that Stage B consumes.

```bash
python -m scripts.generate_pool --smoke         # ~30 s, 4 completions
python -m scripts.generate_pool --seed 0        # ~2-3 h, one seed
python -m scripts.generate_pool --all-seeds     # ~14 h, the full sweep
# Output: experiment_output/raw_runs/logprob/pools/
#           pool_meta_qwen2.5_7b_N32.json
#           pool_qwen2.5_7b_seed{42,123,456,789,0}_N32.jsonl
```

Stage A is the only step that needs Ollama (`qwen2.5:7b` via the
OpenAI-compatible `/v1/chat/completions` endpoint with `logprobs:true`).
Re-runs at the same seeds are NOT byte-identical — `llama.cpp` build and
hardware perturb the per-token logprobs — so re-runs reproduce the reported
effects rather than exact records (manuscript §exp-compute).

### Stage B: Best-of-N selection (540 per-config CSVs from the pools)

Reads the pool jsonls + applies the oracle payoff
$V_i = -w_C(r_i - y_i)^2 + w_A \mathbf{1}\{r_i \geq r_{\min}\}$ at every
$(N, w_A/w_C, r_{\min}, \text{seed})$ cell, argmax-selects with first-index
tie-breaking, and writes 540 per-config CSVs. Pure-deterministic; runs in
~5 s against the pre-generated pools.

```bash
python -m scripts.select_from_pool --all --out-dir /tmp/regen \
    --verify-against-shipped
# Output: 540 CSVs under /tmp/regen, matching the archived
#         experiment_output/raw_runs/logprob/results/*.csv byte-for-byte
#         when run against the archived pools. (Pools and per-config CSVs
#         are retained locally, NOT shipped in the public repo — see the
#         repository structure below — so this byte-equality check applies
#         to the authors' archive or to a locally regenerated Stage-A run.)
```

### Stage 3: Analysis and figures

```bash
# Rebuild hypothesis_results.json end-to-end from raw CSVs
python -m scripts.regenerate_hypothesis_results
# Output: experiment_output/analysis/hypothesis_results.json (rewritten in place)

# Plot H3 achievable-region convexity violation rate by N (descriptive)
python -m scripts.plot_h3_convexity_by_N
# Output: experiment_output/analysis/figures/h3_convexity_by_N.{pdf,png}
```

`N_BOOT` controls bootstrap-CI resamples for H1/H4/H5/H6 (default 10000,
matching the manuscript):

```bash
N_BOOT=2000 python -m scripts.regenerate_hypothesis_results  # ~1 min, smoke
N_BOOT=10000 python -m scripts.regenerate_hypothesis_results # ~5 min, paper
```

Plot script: seconds.

After Stage 3, the `hypothesis_results.json` keys should match Table 1 of
the manuscript. H1, H2, H4, H5, H6 are the **five hypothesis
tests** (all confirmed); **H3 is reported as the descriptive
surface-geometry analysis** of the achievable-(H, C, A) region, not a
confirmed test:

| Hypothesis                                        | $p$-value                | Effect size   |
| ------------------------------------------------- | ------------------------ | ------------- |
| H1 Fixed-axis gating degradation                  | $4.67 \times 10^{-19}$   | $d = 1.10$    |
| H2 Monotone inflation trend (Jonckheere–Terpstra) | $8.49 \times 10^{-5}$    | $\rho = 0.89$ |
| H3 Achievable-region convexity (descriptive)      | binomial test, 10% < 15% | —             |
| H4 Threshold clustering                           | $< 10^{-3}$              | $z = 30.02$   |
| H5 Binding-state specificity                      | $< 10^{-3}$              | $d = 5.32$    |
| H6 Control ($w_A = 0$)                            | $1.35 \times 10^{-23}$   | $d = 1.31$    |

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
  logprob_confidence.py                      # logprob-confidence geomean (manuscript Eq.)
configs/
  params.yaml                                # weight grid, seeds, r_min
scripts/
  run.py                                     # --phase0 branch: load-bearing Stage-0 driver;
                                             #   non-phase0 branch: LEGACY verbalized-era
                                             #   pipeline, NOT the manuscript's experiment
  generate_pool.py                           # Stage A: 32-completion pool jsonl per (task, seed)
  select_from_pool.py                        # Stage B: Best-of-N argmax → 540 per-config CSVs
  regenerate_hypothesis_results.py           # rebuild JSON from per-config CSVs + Phase 0
  plot_h3_convexity_by_N.py                  # H3 stratified-by-N figure
  plot_model_points.py                       # Appendix cross-model figure
  eval_logprob.py                            # one-cell driver for the cross-model figure
  generate_tasks.py                          # task generation
src/
  orchestrator.py                            # per-config runner (legacy verbalized path)
  ollama_client.py                           # Ollama native /api/generate client
  ollama_logprob_client.py                   # Ollama OpenAI-compat /v1 client w/ logprobs
  parser.py                                  # response parser (CONFIDENCE + ANSWER)
  scorer.py                                  # composite payoff
tasks/                                       # 100 tasks (arith/factual/code)
tests/                                       # pytest unit tests
experiment_output/
  analysis/                                  # canonical results (paper's Table 1)
    hypothesis_results.json                  #   shipped; rewritten by Stage 3
    figures/h3_convexity_by_N.{pdf,png}      #   shipped; manuscript Figure 2
    (aggregate metric CSVs regenerated here by Stage 3)
  competence_probe/figures/model_points.*    # shipped; cross-model figure
  raw_runs/                                  # pools + per-config CSVs NOT shipped (gitignored,
                                             #   L73); ONLY the three phase0_calibration.csv
                                             #   files below are shipped
    logprob/pools/                           #   Stage A output (regenerated by --all-seeds)
    logprob/results/                         #   Stage B output (regenerated from pools)
    qwen_2.5/phase0_calibration.csv          #   SHIPPED Phase-0, primary (binding 71/72/74)
    logprob-gemma2_9b/phase0_calibration.csv #   SHIPPED Phase-0, E-B replication (59/61/65)
    logprob-yi_9b/phase0_calibration.csv     #   SHIPPED Phase-0, E-B replication (84/88/97)
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

## Note on Verbalized Confidence

During experiments, we asked the model (here Qwen-2.5-7B) to state its confidence for each answer to help analyze the calibration tradeoff. However, the reported confidence was unreliably high -- the model consistently responded with 100% confidence regardless of correctness or task difficulty. This made meaningful calibration analysis impossible, which motivated our switch to token-level log probabilities as the confidence measure instead.

The per-config CSVs from that earlier verbalised run are retained in the authors' local archive under `experiment_output/raw_runs/qwen_2.5/` (gitignored, not shipped in the public repository; that directory's tracked README documents the provenance). Their degeneracy statistics: selected confidence in the `r_selected` column is r = 1.0 for 93.3% of the 53,928 selected completions and r = 0.95 for the remaining 6.7%. The manuscript's hypothesis results rest on the log-probability pipeline alone; the verbalised run is documented as degenerate, not used as a robustness check.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20392749.svg)](https://doi.org/10.5281/zenodo.20392749)

Archival snapshots are deposited at Zenodo: concept DOI
[10.5281/zenodo.20392749](https://doi.org/10.5281/zenodo.20392749) (always resolves
to the latest version); this release (v1.1-jmlr-submission) is
[10.5281/zenodo.20646964](https://doi.org/10.5281/zenodo.20646964).


```bibtex
@misc{loven2026trilemma,
  title         = {The Behavioral Credibility Trilemma: When Calibrated Autonomy
                   Becomes Impossible},
  author        = {Lov{\'e}n, Lauri and Do, Nam and Mehmood, Hassan and
                   Sah, Dinesh Kumar and Tarkoma, Sasu},
  year          = {2026},
  eprint        = {2605.25739},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.25739}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

Issues and questions: please open a GitHub issue.
