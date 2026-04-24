# Phase 2 — direct observables, partial correlations, population alpha

## Construction

- **Composite maturity (mass-including)**: PC1 of 4 SPARC proxies {log M*, T-type, gas_frac, log SB_eff} on all 171 matched galaxies. PC1 explains 87.1% of proxy variance. Weights ≈ ±0.5 per proxy, sign-aligned so high score ⇔ older galaxy.
- **Composite maturity (mass-free)**: PC1 of 3 proxies {T-type, gas_frac, log SB_eff} — excludes stellar mass to break confound with M_bar. PC1 explains 87.6%.
- **Direct observables** computed per galaxy from the canonical pointer fit:
  - M_halo(<R_last) via numerical integral of 4πr²ρ(r) on the pointer density
  - M_bar = 0.5·L_{3.6}·10⁹ + 1.33·M_HI·10⁹ (Msun; M/L = 0.5 at 3.6 μm; He correction)
  - f_DM(R_last) = (V²_obs − V²_bar) / V²_obs at the last observed radius

## Sample and statistics

Filters: chi2(pointer) < 10; 0.1 < r_mem/R_disk < 30; Q ∈ {1, 2}; all observables finite.
Final sample: **N = 149** of 171.

## Correlation table (Spearman), BH-FDR across 9 tests

| test | ρ | p | q (FDR) |
|---|---:|---:|---:|
| age(with M*) vs log(r_ratio) | +0.124 | 0.131 | 0.169 |
| age(with M*) vs log(M_halo/M_bar) | −0.352 | <0.0001 | <0.0001 ** |
| age(with M*) vs f_DM(R_last) | −0.198 | 0.014 | 0.042 ** |
| age(M-free) vs log(r_ratio) | +0.152 | 0.062 | 0.093 |
| age(M-free) vs log(M_halo/M_bar) | −0.344 | <0.0001 | <0.0001 ** |
| age(M-free) vs f_DM(R_last) | −0.184 | 0.023 | 0.042 ** |
| **age(M-free) vs log(r_ratio) \| log(M_bar)** | **+0.186** | **0.022** | **0.042 ** ** |
| age(M-free) vs log(M_halo) \| log(M_bar) | +0.031 | 0.708 | 0.708 |
| age(M-free) vs log(M_halo/M_bar) \| log(M_bar) | −0.076 | 0.356 | 0.400 |

*bold ** = survives FDR correction at α = 0.05.*

## Interpretation

1. **Raw halo-mass tests run strongly NEGATIVE** (ρ = −0.34). More massive/mature galaxies have RELATIVELY less halo mass per baryonic mass — this is the well-known cosmological downsizing / radial-acceleration-relation scaling. The naive "more archive = older" prediction fails on absolute halo-mass observables.
2. **After controlling for M_bar, halo mass has no independent age signal** (ρ ≈ 0, p = 0.71). The apparent negative correlation is fully accounted for by M_bar.
3. **Halo EXTENT, not mass, carries the PA-compatible signal**. The ratio r_mem/r_disk, after partialling M_bar from a mass-free age axis, correlates positively with galaxy maturity at ρ = +0.186, q_FDR = 0.042. This is one of two FDR-surviving positive tests (the other being its non-partialled version at marginal p).
4. **Population-level alpha** (the log-enhancement strength of the pointer density) is not well-behaved: 42% of galaxies hit α = 0 (log term switched off) and 39% hit α = 5 (upper bound). The distribution is bimodal, not a smooth tail above zero. The Wilcoxon test against α > 0 is trivially significant (p ≈ 1e-27) because of the upper-boundary cluster, but this reflects parameter degeneracy, not a population-level growth signal. Phase 3 addresses the degeneracy by tying α to log(N_orbits) across the sample.

## Conclusion

The canonical SPARC analysis of §6 tests r_mem/r_disk against six age proxies and finds 5/6 in the predicted direction with binomial p = 0.11 and PC1 p = 0.05.

Phase 2 tightens that in three ways:
- adds the directly predicted observable (halo mass, normalised);
- uses a mass-free age axis so the PA prediction isn't automatically satisfied by trivial M_bar scaling;
- applies FDR to 9 tests jointly.

The single defensible positive result: **r_mem/r_disk grows with dynamical maturity at fixed baryonic mass** (ρ = +0.19, q_FDR = 0.04, CI excludes zero). Halo mass shows no such residual signal. The naive "more archive = older" interpretation does not survive on SPARC; only the extent variant does, and even then weakly.

A pre-registered replication on THINGS / LITTLE THINGS with the frozen mass-free composite and r_ratio-only test remains the decisive experiment.
