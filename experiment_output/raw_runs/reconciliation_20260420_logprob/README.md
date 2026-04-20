# Reconciliation Data Package (2026-04-20)

This directory is a separated analysis package for the updated/reordered
hypothesis report and supervisor review.

## What this package contains

- `analysis_snapshot/`
  - `hypothesis_results.json`
  - `per_config_metrics.csv`
  - `inflation_metrics.csv`
  - `brier_decomposition.csv`
  - `pairwise_effect_sizes.csv`
- `tables_snapshot/`
  - `hypothesis_results.tex`
  - `hypothesis_reconciliation.tex`
  - `summary_metrics.tex`
  - `effect_sizes.tex`
  - `brier_decomposition.tex`
- `findings_summary.csv`
  - Compact table of pre-registered, supplementary, and post-hoc findings.
- `consistency_check.json`
  - Math-consistency crosswalk against main-paper results/claims.
- `run_manifest.json`
  - Commands used to generate analysis and tables.

## Provenance

- Source experimental results: `expirement_results/logprob/results/`
- Source analysis output: `analysis_logprob/`
- Source tables: `reports/tables_logprob/`

No new model-generation experiment was run for this package.
This is a reproducible reconciliation snapshot from the existing logprob run.
