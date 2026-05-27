# Legacy: verbalized-confidence run

This directory contains results from an earlier run in which the per-completion
confidence `r` was elicited *verbally* — the model was asked to state its own
confidence in prose, and the value was parsed from the response.

**These results are superseded by the log-probability-based run at
`../logprob/`.** The log-probability confidence (geometric mean of per-token
probabilities) is the primary metric used by the manuscript's analysis
pipeline and by every `regenerate_hypothesis_results.py` / `plot_*` script in
`scripts/`.

The directory is retained for audit-trail purposes only:

- It documents the methodological evolution of the confidence metric.
- It allows a reader to confirm that the verbal-confidence and
  log-probability-confidence runs are not conflated.

Do not use these CSVs for the paper's analysis. If you want to reproduce the
manuscript's numbers, use `../logprob/results/` exclusively.
