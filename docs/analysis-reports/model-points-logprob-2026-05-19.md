# Real models as points — logprob cross-model placement (2026-05-19)

Source: `analysis.model_points` LOGPROB loader
(`all_logprob_model_coords`, committed `7ee9415`) over
`experiment_output/logprob_xmodel/<model>/<model>_s<seed>.csv`
(7-col schema `task,category,seed,r_logprob,answer,y,acted`). Figure
written by `scripts/plot_model_points.py` (mode `logprob`,
`build_logprob_figure`) to
`experiment_output/analysis/figures/model_points_logprob.{pdf,png}`.

This is a **separate** report from `model-points-2026-05-19.md` (the
competence-probe path); that report is not superseded or overwritten.

## Honest framing (verbatim from `analysis.model_points.HONEST_CAPTION_LOGPROB`)

> Descriptive placement of real, named open-weight models on the behavioral
> axes using the logprob-confidence path; this is NOT a trilemma proof, an
> impossibility result, or a traced trade-off region. The autonomy axis A is
> the answer-commitment rate: a task counts as acted only when a valid
> answer was successfully parsed from the completion via the ANSWER-parser,
> and as 'abstained' otherwise. This answer-parse acted predicate conflates
> deliberate deferral with mere parse-fail (unparseable-output) responses,
> so A measures behavioral answer-commitment, not
> autonomy-as-chosen-deferral. The logprob path runs ungated at N=1 (no
> argmax, no abstention incentive, no gating reward), so A is design-pinned
> by construction, not a traced trade-off. Consequently the (H, C, A) panel
> is competence-confounded and cannot be read as a causal trade-off: model
> placement reflects task competence, not a mechanism. The causal trilemma
> claim rests on the gated mechanism experiment (hypotheses H1/H2/H4/H5)
> together with the theory, NOT on this scatter. The calibration value C
> uses the per-output logprob-confidence; this loader is an independent
> reimplementation of the manuscript logprob-confidence equation and is NOT
> bit-verified against the original 540-config run, so C should be read as a
> faithful reimplementation, not a byte-exact reproduction.

The caveats above are **not burned onto the figure image**. Both prior
on-figure caption boxes (the `fig.text(...)` honest-caption block and the
`_panel_placement` "A ≈ 1.0 (ungated probe)" textbox) have been removed; the
disclosures live here and, later, in the LaTeX figure caption. The figure
carries only a short descriptive title and a theory-reference annotation
(see below).

## What this figure is — and is NOT

- It is a **descriptive placement** of named open-weight models on the
  behavioral (H, C) plane with A encoded as marker size. It is **NOT** a
  trilemma proof, **NOT** an impossibility result, and **NOT** a traced /
  achievable trade-off region or Pareto frontier.
- **A is the answer-commitment rate** computed via the ANSWER-parser
  (`acted == 1` iff a valid answer was parsed from the completion). It
  therefore **conflates deliberate deferral with parse-fail**
  (unparseable-output) responses: a model that intentionally declines and a
  model that emits an unparseable answer are both scored "abstained". A is
  behavioral answer-commitment, not autonomy-as-chosen-deferral. (This is
  the logprob-path wording of the construct-validity caveat; it reuses the
  `HONEST_CAPTION_LOGPROB` text.)
- **A is design-pinned, not a trade-off.** The logprob path runs **ungated
  at N=1** (no argmax, no abstention incentive, no gating reward). Any
  apparent (or absent) autonomy structure is an artifact of the design, not
  evidence of a mechanism.
- **(H, C, A) is competence-confounded.** A model sits higher on H/C because
  it is more competent on these tasks, not because of an
  autonomy/helpfulness/calibration mechanism. **The causal trilemma claim
  rests on the gated mechanism experiment (H1/H2/H4/H5) together with the
  theory, NOT on this scatter.**
- **C is an independent reimplementation.** C is computed from the
  per-output logprob-confidence using an independent reimplementation of the
  manuscript logprob-confidence equation. It is **NOT bit-verified** against
  the original 540-config run. Read C as a faithful reimplementation, not a
  byte-exact reproduction.

## Transport / template disclosure

Per-token logprobs are available from Ollama **only** via the
`/v1/chat/completions` endpoint, which applies the model's **chat
template** to the prompt. The original archived run used Ollama's native
`/api/generate` endpoint (no chat template). This figure therefore forces
`/v1/chat/completions` because logprobs are otherwise unavailable. The
forced endpoint is applied **uniformly to all figure models through one
common client**, so the three points are comparable to each other **by
construction** (one client, one regime, within-figure). This transport /
template difference is part of the disclosed *figure ≠ Table* "different
client + regime, by design" caveat — it is **not** a separate, unhandled
confound.

## qwen-2.5-7B: figure ≠ Table, BY DESIGN

The figure's qwen-2.5-7B point comes from the **new client at the N=1
*ungated* 100-task regime**. It will **NOT** numerically match the paper's
qwen-7B H1–H6 / Table results, which come from the **540-config *gated*
regime under the original client**. This mismatch is **intentional**:
qwen-7B appears here as a **cross-model comparability anchor under one
common client/regime**, not as a reproduction of the Table. This must be
flagged in the eventual **A1 figure caption** and the **I3 appendix**:
state explicitly that the figure's qwen-7B value is a within-figure anchor,
not a Table reproduction, so no reader infers a numeric discrepancy where
none is claimed.

## Figure layout — single-panel τ-trajectory scatter

The placement panel is a **single-panel (H, C) scatter** with one
trajectory per model:

- each model contributes a sequence of points across a τ-sweep
  (calibration-threshold sweep); the **baseline** point sits at the
  bottom-right end of the trajectory and points move **up-and-left** as τ
  increases (higher calibration, lower helpfulness);
- A (answer-commitment rate) is encoded as **marker size**;
- segments with `n_acted < 10` are gated out of the rendering to suppress
  small-sample spikes (commit `a92066c`);
- 8 models appear in the figure (including the late-added
  `command-r7b`).

No corner star, no fabricated achievable region, and no fake Pareto
frontier are drawn — this is purely descriptive within-figure placement.

## (H, C, A) ± 95% bootstrap CI per model

| model | H [95% CI] | C [95% CI] | A [95% CI] | n_acted | n_tasks | n_seeds | partial |
|---|---|---|---|---|---|---|---|
| gemma2_9b | 0.398 [0.354, 0.440] | 0.556 [0.523, 0.588] | 0.996 [0.990, 1.000] | 498 | 500 | 5 | False |
| mistral_7b-instruct-q4_K_M | 0.204 [0.168, 0.240] | 0.648 [0.617, 0.674] | 0.790 [0.754, 0.826] | 395 | 500 | 5 | False |
| qwen2.5_7b | 0.294 [0.254, 0.334] | 0.437 [0.402, 0.471] | 1.000 [1.000, 1.000] | 500 | 500 | 5 | False |

Numbers were filled from
`analysis.model_points.all_logprob_model_coords` run on
`experiment_output/logprob_xmodel/` after the L.4 real run completed
(3 models × 5 seeds × 100 tasks = 1500 task rows; every per-seed CSV
verified at exactly 100 rows via `csv.DictReader`). Bootstrap config:
`random_state=0`, B = 2000, percentile 2.5 / 97.5 (same engine as the
competence path, reused via the `point_fn` hook). All three models are
complete (5 seed CSVs each, 100 rows per CSV) so no partial-flag fading
is applied; partial-model semantics remain available for future runs.

Descriptively: the three models fall in distinct regions of the (H, C)
plane. gemma2_9b has the highest helpfulness (H = 0.40) and an
intermediate calibration value (C = 0.56), with A pinned at the top end
(0.996). mistral_7b-instruct-q4_K_M has the lowest helpfulness
(H = 0.20) but the highest calibration value (C = 0.65), and is the only
model with appreciably sub-1.0 answer-commitment (A = 0.79, i.e. 105 of
500 rows did not yield a parseable answer — these are parse-fails under
the answer-parse `acted` predicate, not deliberate deferrals). qwen2.5_7b
sits between the two on helpfulness (H = 0.29) and at the lowest
calibration value (C = 0.44), with A = 1.00. Per the
`HONEST_CAPTION_LOGPROB` caveats above, this is **descriptive
cross-model placement under one common client/regime only** — it is not
an impossibility, a trade-off region, a Pareto frontier, or a
reproduction of any Table; mistral's lower A specifically reflects
answer-parse failures, not a measured deferral mechanism.

The rendered figure is at
`./model-points-logprob-2026-05-19.png` (relative to this report) and
also placed in the JMLR A1 manuscript figures directory as
`figures/model-points-logprob-2026-05-19.png` (+ `.pdf`).

## Provenance

- Code: `scripts/plot_model_points.py`, function `build_logprob_figure`
  (Agg backend forced before pyplot import; I/O parameterised; no network,
  no model calls). Both on-figure caption boxes removed.
- Coordinates: `analysis.model_points.all_logprob_model_coords` at
  `7ee9415` (answer-parse `acted` predicate; bootstrap reuses
  `_bootstrap_ci` + `_pct` via the `point_fn` hook).
- Caption SSoT: `analysis.model_points.HONEST_CAPTION_LOGPROB`
  (enforced by `tests/analysis/test_logprob_xmodel_metrics.py::
  test_honest_caption_logprob_caveats`).
- Tests: `tests/analysis/test_model_points_figure.py` (logprob-mode
  block: loader-sourced, no-burned-in-caption-box, Agg backend,
  competence-path regression guard).

The figure's `y` column was reanalyzed in-place by
`scripts/reanalyze_logprob_xmodel.py` using `analysis.robust_verify`
(charitable cross-model answer extraction). `.csv.orig` backups
preserve the pre-reanalysis state for the 6 originally-evaluated
models. The 2 later-added models (`deepseek-llm:7b-chat` and
`command-r7b`) were evaluated after `robust_verify` was already
integrated into the eval-time path, so they used `robust_verify`
natively and no `.csv.orig` backup was produced for them. This
asymmetry is provenance, not a defect.
