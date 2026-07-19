# Note: the aggregate CSVs in this directory are VERBALISED-ERA (superseded)

The following untracked CSVs are aggregates of the **superseded
verbalised-confidence run** (see the repository README's "Note on
Verbalized Confidence" and `../raw_runs/qwen_2.5/README.md`), retained
locally for audit-trail purposes only:

- `per_config_metrics.csv` — e.g. mean Brier at N=1 is 0.7184 here
  (verbalised era: r ≈ 1.0, so Brier ≈ 1 − accuracy) vs the logprob
  run's 0.5511. **Do not compare against the shipped JSONs.**
- `brier_decomposition.csv` — regenerable for the logprob run via the
  maintained `analysis/metrics.py::compute_brier_decomposition_by_config`
  over `../raw_runs/logprob/results/`.
- `inflation_metrics.csv`, `pairwise_effect_sizes.csv`,
  `summary_mean_{A,H,brier}_r{0.5,0.7,0.9}.csv` — same era.

**Canonical (manuscript) artifacts in this directory are the tracked
files only:** `hypothesis_results.json` (Table 1) and its sibling
suites (`-comonotone`, `-rearrangement`, `-gemma2_9b`, `-yi_9b`,
`-random`, `-proxy`), `comonotone_violation.json`, `ea-comparison.md`,
`controls-comparison.md`, and `figures/`. All of these are computed
from the **log-probability** pipeline. `figures.py`, `metrics.py`, and
`hypothesis_tests.py` here are frozen snapshots (gitignored; never
imported) — the maintained package is the repo-root `analysis/`.

Do not use the verbalised-era CSVs for the paper's numbers. This note
follows the same pattern as the m-B-1 annotation in
`../raw_runs/logprob/results/README.md` (2026-06-11): the files are
annotated, not deleted, to preserve provenance.
