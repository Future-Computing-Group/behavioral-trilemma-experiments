# Real models as points — descriptive placement (2026-05-19)

Source: `analysis.model_points` (committed `ca59ce2`) over
`experiment_output/competence_probe/_runs`. Figure written by
`scripts/plot_model_points.py` to
`experiment_output/competence_probe/figures/model_points.{pdf,png}`.

## Honest framing (verbatim from `analysis.model_points.HONEST_CAPTION`)

> Descriptive placement of real, named open-weight models on the behavioral
> axes; this is NOT a trilemma proof, an impossibility result, or a traced
> trade-off region. The autonomy axis A is approximately 1.0 for every model
> because the competence probe ran ungated (no abstention incentive / no
> gating reward), so A is pinned by construction and the apparent absence of
> an autonomy trade-off is an artifact of the design, not evidence.
> Consequently the (H, C, A) panel is competence-confounded and cannot be
> read as a causal trade-off: model placement there reflects task
> competence, not a mechanism. The causal trilemma claim rests on the gated
> mechanism experiment (hypotheses H1/H2/H4/H5) together with the theory,
> not on this scatter. What the calibration panel does show is each model's
> per-output reliability structure (reported confidence r versus realized
> correctness), i.e. the direction and magnitude of per-model
> overconfidence.

This report makes no claim that the figure establishes the trilemma; the
figure is a descriptive snapshot only. The causal account lives in the gated
mechanism experiment (H1/H2/H4/H5) and the theory.

## (H, C, A) ± 95% bootstrap CI per model

| model | H [95% CI] | C [95% CI] | A [95% CI] | n_acted | n_tasks | n_seeds | partial |
|---|---|---|---|---|---|---|---|
| gemma2_9b | 0.388 [0.344, 0.430] | 0.392 [0.348, 0.435] | 1.000 [1.000, 1.000] | 500 | 500 | 5 | no |
| mistral_7b-instruct-q4_K_M | 0.226 [0.188, 0.262] | 0.262 [0.225, 0.299] | 0.920 [0.896, 0.944] | 460 | 500 | 5 | no |
| qwen2.5_7b | 0.276 [0.238, 0.314] | 0.283 [0.244, 0.322] | 0.990 [0.980, 0.998] | 495 | 500 | 5 | no |
| qwen2.5_32b | 0.380 [0.315, 0.450] | 0.381 [0.316, 0.450] | 1.000 [1.000, 1.000] | 200 | 200 | 2 | **yes (partial: 2 seeds)** |

`qwen2.5_32b` is flagged `partial` (only 2 of 5 seeds; n_acted = 200). It is
rendered faded/hollow with a "partial: 2 seeds" label and is never averaged
into the figure as if it were a complete point.

## Per-model counts and flags

- gemma2_9b — n_acted 500 / 500, 5 seeds, complete.
- mistral_7b-instruct-q4_K_M — n_acted 460 / 500, 5 seeds, complete (40
  abstentions/parse-fails account for A = 0.920).
- qwen2.5_7b — n_acted 495 / 500, 5 seeds, complete.
- qwen2.5_32b — n_acted 200 / 200, 2 seeds, **partial** (incomplete seed
  set; treat point and CI as provisional).

## Reading the panels honestly

A is pinned at approximately 1.0 for every model because the probe ran
ungated: there was no abstention incentive and no gating reward, so models
acted on essentially every task. The (H, C) plane is therefore
competence-confounded — a model sits higher on H/C because it is more
competent on the probe tasks, not because of any autonomy/helpfulness/
calibration mechanism. Panel 2 encodes A as marker size and annotates
"A ≈ 1.0 (ungated probe)" precisely so the pinned axis is visible rather
than silently traced as a region.

The calibration panel (Panel 1) is the substantive one: it plots, per model,
the binned reliability curve (mean correctness vs reported-confidence bin)
against the y = x ideal line, with the raw per-output points faint behind
it. For these 4 models the reliability curves sit well below the diagonal
across the confidence range, i.e. every model is systematically
overconfident (reported confidence exceeds realized correctness), with
mistral_7b the most overconfident and gemma2_9b / qwen2.5_32b the least.

## Provenance

- Code: `scripts/plot_model_points.py` (Agg backend; I/O parameterised).
- Coordinates: `analysis.model_points` at `ca59ce2`, bootstrap
  `random_state=0`, B = 2000, percentile 2.5 / 97.5.
- Tests: `tests/analysis/test_model_points_figure.py`,
  `tests/analysis/test_model_points.py::test_honest_caption_exists_and_is_honest`.
