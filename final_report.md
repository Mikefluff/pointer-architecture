# Pointer Architecture: canonical analysis on SPARC

Galaxies fitted: 171 / 175 (Table 1 + rotmod matched).

## Fit quality, head-to-head

| Model | median chi2 | frac chi2<3 | median AIC | median BIC |
|---|---:|---:|---:|---:|
| pointer | 0.801 | 81% | 18.20 | 23.59 |
| nfw | 1.167 | 77% | 21.60 | 24.18 |
| burkert | 0.598 | 87% | 14.19 | 17.43 |

AIC best model split across 171 galaxies: pointer=13, nfw=55, burkert=103.

## Cross-validation (3-fold on radial points, pointer model)

- pointer: train chi2 median 13.52, test chi2 median 41.56, test/train ratio median 3.01
  A ratio near 1 suggests no overfitting; >>1 is a red flag.

## Correlations on log(r_mem/r_disk), pointer model

Sample: N = 122 (chi2<10, 0.1<ratio<30, Q=1,2, all proxies finite).

| proxy | exp | r | p | p_perm | spearman | ci_lo | ci_hi | dir |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T | NEG | -0.206 | 0.021 | 0.023 | -0.212 | -0.364 | -0.039 | ok |
| log_Mstar | POS | +0.139 | 0.123 | 0.128 | +0.130 | -0.060 | +0.324 | ok |
| gas_frac | NEG | -0.069 | 0.445 | 0.447 | -0.052 | -0.242 | +0.119 | ok |
| log_N_orbits | POS | +0.432 | 0.000 | 0.000 | +0.427 | +0.254 | +0.594 | ok |
| concentration | POS | -0.002 | 0.981 | 0.984 | -0.041 | -0.148 | +0.135 | WRONG |
| log_SBeff | POS | +0.275 | 0.002 | 0.002 | +0.291 | +0.130 | +0.412 | ok |

Directional summary: 5/6 in predicted direction.
Binomial test (no post-hoc exclusions): p = 0.1094.

## PCA on 6 proxies

| component | var explained | r(PC,log ratio) | p_perm | spearman | ci |
|---|---:|---:|---:|---:|---|
| PC1 | 60.7% | +0.177 | 0.053 | +0.182 | [+0.01,+0.34] |
| PC2 | 19.0% | -0.394 | 0.000 | -0.392 | [-0.55,-0.22] |
| PC3 | 15.1% | +0.165 | 0.071 | +0.170 | [-0.02,+0.35] |

PCA directional test: PC1 sign-aligned to log(M*) loading (older-more-massive direction).
A positive PC1~log-ratio correlation supports memory-accumulation hypothesis.

Run duration: 44.8 min.
