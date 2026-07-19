# E-C controls comparison: oracle (primary) vs uniform-random vs no-oracle proxy

Generated 2026-07-19 from the two canonical N_BOOT=10000 control artifacts
(`hypothesis_results-random.json`, `hypothesis_results-proxy.json`) against
the primary `hypothesis_results.json`; protocol = EXPERIMENT-PLAN §11
amendment 3 (E-C), design frozen before either control ran. Same archived
Qwen pools, same Phase-0 inputs (binding sets 71/72/74), same Holm-5
family {H1, H2, H4, H5, H6}; H3 descriptive, outside the family. These
are the manuscript's two declared controls (uniform random selection,
§exp-controls; no-oracle proxy, §exp-protocol), materialized post hoc as
Stage-B re-analyses of the frozen pools (zero model calls; audit
LB-1/LB-2, 2026-07-19).

## Holm-5 outcomes

| Hypothesis | primary (oracle) | random | proxy |
|---|---|---|---|
| H1 | t=10.95, p=4.7e-19, **reject** | t=-0.20, p=0.58, no | t=NaN (0/0 contrast), no |
| H2 | z=3.76, p=8.5e-05, **reject** | z=-0.48, p=0.68, no | z=0 (constant in w), no |
| H4 (family p = min over r_min) | z=30.16 at 0.7, **reject** | z=0.49 at 0.7, p=0.31, no | z=-15.95 at 0.7; family rejects via r_min=0.9 only (top-of-range mass) |
| H5 | t=208, d=5.35, **reject** | d=6.47, reject | d=10.15, reject |
| H6 | t=-13.08, p=1.3e-23, **reject** | t=-0.83, p=0.20, no | t=+2.78, p=0.997, no |

## Reading (pre-stated in E-C.4; realized outcomes as-is)

- **Uniform random (the "no optimization pressure" control):** every
  optimization-pressure signature vanishes, exactly as pre-stated. At
  N=32, r_min=0.7 the binding-task threshold-window mass is 19.2% vs the
  base arm's 18.6% (z = 0.49, p = 0.31) — against 65.8% under oracle
  selection; the H1 weight contrast is d = -0.02 (p = 0.58); H2 shows no
  trend; H6 shows no improvement (selection is payoff-blind, mean ΔBS =
  -0.005, p = 0.20). **H5 remains significant (d = 6.47) — disclosed in
  E-C.4 as expected:** H5 contrasts binding vs non-binding *task
  classes*, not selection arms, so under random selection it measures the
  base completion law's overconfidence on hard tasks, not
  selection-driven inflation. The systematic inflation of the primary
  run does not survive the removal of the payoff-maximising mechanism.
- **No-oracle proxy (the oracle/no-oracle-gap control):** the proxy is
  *not* oracle-like on this data, in the way the manuscript's own
  characterization predicts ("incentivizes extreme reports independent
  of the gating term"): it selects the extremeness argmax (mean selected
  r = 0.958 at N=32, invariant across w and r_min), which is verified
  w-invariant in 9000/9000 (N, r_min, seed, task) groups — so the
  weight-contrast tests H1/H2 are structurally 0/0-degenerate for this
  selector (the same construct signature as the E-A.1a V_c rank-aligned
  limit, disclosed rather than forced), and instead of pooling at the
  gate it overshoots it: binding-task mass in [0.7, 0.8] is 2.5% vs the
  oracle's 65.8%, with the mass piled in [0.9, 1.0] (the r_min=0.9 arm
  is why the H4 family-minimum still rejects). H5 strengthens (d =
  10.15) and H6 reverses (calibration *worsens* with N at w_A = 0,
  t = +2.78), both consequences of y-blind extremeness selection.

## Oracle/no-oracle gap (headline threshold r_min = 0.7, N = 32)

| Quantity | oracle | proxy | gap (proxy - oracle) |
|---|---|---|---|
| Selected-completion mean Brier, w_A = 0 | 0.2739 | 0.6028 | +0.3289 |
| Selected-completion mean Brier, pooled w > 0 | 0.3529 | 0.6028 | +0.2499 |
| Selected-completion mean Brier, w_A/w_C = 4 | 0.3677 | 0.6028 | +0.2351 |
| Binding-task mass in [0.7, 0.8], N=32, w > 0 | 0.658 | 0.025 | -0.633 |
| Mean selected r, w > 0 | 0.834 | 0.958 | +0.124 |
| Helpfulness H (fraction correct), w > 0 | 0.383 | 0.320 | -0.063 |

(Per-completion means over the 540-grid cells at N=32, r_min=0.7; base
arm reference: N=1 binding-task window mass 0.186.)

## Artifacts

- `hypothesis_results-random.json`, `hypothesis_results-proxy.json`
  (canonical, N_BOOT=10000; tracked).
- Per-config CSVs: `raw_runs/logprob/results-random/`,
  `raw_runs/logprob/results-proxy/` (540 each; gitignored per L73,
  regenerable deterministically: `python -m scripts.select_from_pool
  --all --selector {random,proxy} --out-dir ...` — the random selector
  is seeded from the config fields, so re-runs are byte-stable).
