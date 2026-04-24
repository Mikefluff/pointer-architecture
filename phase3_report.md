# Phase 3 — Constrained Pointer Architecture

## Population relations

Using 6 galaxies with free alpha not pinned at bounds:

- alpha  ≈ +6.303 + -2.330 · log10(N_orbits)   (r=-0.304, RMSE=1.169)
- r_core ≈ +7.081 · R_disk   (r=-0.251, RMSE=28.131)

## Refit with alpha, r_core fixed (k=4 per galaxy)

- median chi2 (constrained pointer) = 0.621
- median AIC  (constrained pointer) = 14.891

AIC best model, head-to-head at k=4:

- Pointer (constrained): 60
- NFW: 54
- Burkert: 57

Compare to the free-parameter canonical run: pointer wins AIC on 13/171 with k=6. The constrained variant matches Burkert and NFW in parameter count.
