# E-A 3-way comparison: primary (oracle Eq. 11) vs V_c (E-A.1a) vs rearrangement (E-A.1b)

Generated 2026-06-12 from the three canonical N_BOOT=10000 artifacts
(`hypothesis_results.json`, `hypothesis_results-comonotone.json`,
`hypothesis_results-rearrangement.json`); protocol = EXPERIMENT-PLAN
§11.E-A as amended by §11 E-A amendment 1 (2026-06-12). Same archived
Qwen pools, same Phase-0 inputs (binding sets 71/72/74), same Holm-5
family {H1, H2, H4, H5, H6}; H3 descriptive, outside the family.

| Hypothesis | statistic | primary p | Holm | V_c statistic | V_c p | Holm | rearr. statistic | rearr. p | Holm |
|---|---|---|---|---|---|---|---|---|---|
| H1 | t=11 | 4.67e-19 | reject | t=NaN | NaN | **no** | t=11.7 | 1.24e-20 | reject |
| H2 | z=3.76 | 8.49e-05 | reject | z=0 | 0.5 | **no** | z=4.19 | 1.39e-05 | reject |
| H4 | z=30.2 | 0 | reject | z=-16 | 0 | reject | z=-16.5 | 0 | reject |
| H5 | t=208 | 0 | reject | t=282 | 0 | reject | t=242 | 0 | reject |
| H6 | t=-13.1 | 1.35e-23 | reject | t=1.71 | 0.955 | **no** | t=-8.38 | 1.84e-13 | reject |

Notes:

- p is the value entered into the Bonferroni–Holm 5-family (for H4: the
  minimum over the per-threshold clustering p-values, as in
  `analysis/hypothesis_tests.py::run_all_tests`); Holm = corrected decision
  at alpha = 0.05.
- **Primary:** all five reject (manuscript Table 1).
- **V_c (E-A.1a, rank-aligned limit per amendment 1):** {H4, H5 reject;
  H1, H2, H6 structurally undefined as weight contrasts} — V_c is
  y-independent above the gate, so selection is w-invariant by
  construction (verified: 9000/9000 groups identical across all 6
  w-levels; H1 contrast identically zero, t = 0/0 = NaN; H2 trend
  constant, z = 0; H6 control sign-reversed, the prop:bon(i) inflation
  of comonotone selection alone). Reported as the empirical realization
  of prop:bon(ii)'s noiseless rank-aligned limit, NOT as a robustness
  failure (criterion retracted for V_c in amendment 1).
- **Rearrangement (E-A.1b, robustness check proper):** all five Holm-5
  decisions match the primary -> robustness confirmed (5/5) under the
  pre-stated amendment-1 criterion. The selector satisfies prop:bon(c)
  by construction (comonotone coupling on the cleared set) while
  preserving each pool's score distribution exactly.
- E-A.2 violation rates of the original oracle selector (unchanged by
  amendment 1): event-level 0.6443, pair-level 0.8878
  (`comonotone_violation.json`).
