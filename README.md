# Pointer Architecture — reproducibility pipeline

Four-script pipeline behind the SPARC results in `main.tex`, section 6.

1. **`analysis.py`** — canonical analysis. Fits Pointer, NFW and Burkert
   halos to all 175 SPARC disk galaxies with the same pipeline, reports
   χ², AIC, BIC, cross-validation, 6-proxy correlations and PCA.
2. **`phase2_derived.py`** — direct observables (M_halo, M_bar, f_DM),
   mass-free composite age, partial correlations with BH-FDR, Wilcoxon on α.
3. **`phase3_constrained.py`** — constrained 4-parameter Pointer
   (α, r_core tied to population relations) vs.\ NFW/Burkert at parity.
4. **`phase4_residuals.py`** — structural features of residuals after a
   best-fit Burkert, correlated with composite age under BH-FDR.

## Quick start

```bash
# 1. Python 3.11+ with a clean venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. SPARC data
#    - Download SPARC_Lelli2016c.mrt from http://astroweb.cwru.edu/SPARC/
#    - Download Rotmod_LTG.zip, unzip into Rotmod_LTG/
#    Both must sit next to the scripts.

# 3. Run the canonical pipeline. Expect ~45-60 min.
python analysis.py

# 4. Run the three follow-up analyses (each ~10-20 min).
python phase2_derived.py
python phase3_constrained.py
python phase4_residuals.py
```

## Outputs

Canonical (`analysis.py`):
- `final_fits.csv` — per galaxy × model: χ², AIC, BIC, fit params, pointer
  CV train/test χ²
- `final_corr.csv` — 6-proxy correlation table (Pearson, Spearman, CI, permutation p)
- `final_pca.csv` — PCA of the 6 proxies
- `final_report.md` — human-readable summary

Phase 2 (`phase2_derived.py`):
- `phase2_derived.csv` — M_halo(<R_last), M_bar, f_DM(R_last), composite age
- `phase2_report.md` — partial correlations, BH-FDR, Wilcoxon

Phase 3 (`phase3_constrained.py`):
- `phase3_constrained.csv` — constrained-pointer fits, AIC, BIC
- `phase3_report.md` — AIC head-to-head at k=4

Phase 4 (`phase4_residuals.py`):
- `phase4_residuals.csv` — residual-structure features per galaxy
- `phase4_report.md` — feature correlations with composite age, BH-FDR

A reference set of outputs from the author's run sits alongside the
scripts; re-running should reproduce them up to the second decimal
(differential-evolution seed plus Sobol initialisation is deterministic).

## Methodology notes

- **Physics.** Enclosed mass uses the spherical integral M(<r) = ∫ 4πr'²ρ(r')dr',
  computed numerically with `scipy.integrate.cumulative_trapezoid` on a
  200-point grid. The log-enhancement term is kept **inside** the integrand.
  Halo velocity is V²(r) = G·M(<r)/r with G = 4.30091×10⁻⁶ kpc·(km/s)²/M_⊙.
  No magic unit-conversion constants.
- **Optimization.** Scale parameters (ρ₀, r_mem, r_core) are fit in log₁₀
  space to let differential evolution explore the ~8-order-of-magnitude
  dynamic range uniformly. Initialization uses a Sobol sequence and a
  grid of handcrafted Nelder-Mead restarts.
- **Proxies.** N_orbits is used on a log scale to compress its heavy tail.
  No proxy is excluded post-hoc.
- **Correlation significance.** Binomial and Pearson p-values are reported
  alongside a permutation p (5000 permutations), since the 6 proxies
  are strongly correlated and an independence-assumed binomial would
  overstate significance. The PCA-based test on PC1 sidesteps
  collinearity: PC1 is sign-aligned to the log(M*) direction so that a
  positive PC1~log(r_ratio) correlation supports the
  memory-accumulation hypothesis.
- **Cross-validation.** 3-fold CV on radial points of each galaxy, for
  the Pointer model only. A large test/train χ² ratio flags overfitting
  at the galaxy level.

## Files

- `analysis.py` — canonical pipeline (this is the one to cite from main.tex)
- `requirements.txt` — pinned dependencies
- `README.md` — this file

## Version

Matches the preprint `main.tex` dated April 2026.
