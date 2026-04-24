# Phase 4 — Residual structure after Burkert subtraction

Sample: 171 galaxies refit with Burkert halo; 168 pass chi2<10 cut for the correlation test.

## Feature correlations with composite age (Spearman)

| feature | n | rho | p_perm | q (FDR, BH) |
|---|---:|---:|---:|---:|
| resid_rms | 168 | +0.428 | 0.0000 | 0.0000 |
| resid_mean_abs | 168 | +0.412 | 0.0000 | 0.0000 |
| resid_radial_slope | 168 | +0.215 | 0.0049 | 0.0095 |
| resid_radial_slope_spearman | 168 | +0.122 | 0.1164 | 0.1164 |
| resid_compactness | 168 | +0.153 | 0.0489 | 0.0587 |
| resid_spectral_slope | 140 | +0.228 | 0.0063 | 0.0095 |

Positive composite_age = more evolved galaxy. Features with q_fdr < 0.05 survive multiplicity.
