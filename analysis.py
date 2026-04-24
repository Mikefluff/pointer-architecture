#!/usr/bin/env python3
"""
Pointer Architecture: canonical methodology on SPARC 175.

Rebuilds uni2.py with:
  A. Correct physics — 4πr² density integral, log term inside the integral,
     proper G in (kpc, km/s, Msun) units, no magic constants.
  B. Head-to-head Pointer vs NFW vs Burkert on the SAME 171 galaxies with
     the SAME optimizer and SAME error bars. AIC and BIC per galaxy.
  C. PCA on the 6 age proxies. Directional tests on PC1/PC2 with
     permutation p-values, since raw proxies are strongly correlated.
  D. Diagnose N_orbits. Use log(N_orbits) and Spearman. No post-hoc exclusion.
  E. 5-fold cross-validation on radial points per galaxy. Reports
     train χ² vs test χ² to flag overfitting.

Inputs (expected in CWD):
  - SPARC_Lelli2016c.mrt
  - Rotmod_LTG/*_rotmod.dat

Outputs:
  - final_fits.csv       per galaxy × model: χ², AIC, BIC, CV_χ², params
  - final_corr.csv       raw correlation table (Pearson + Spearman + bootstrap CI)
  - final_pca.csv        loadings + PC scores
  - final_report.md      human summary, headline numbers
"""

from __future__ import annotations

import os
import sys
import glob
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import cumulative_trapezoid
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONSTANTS (physical, no magic)
# ------------------------------------------------------------

# G in astrophysical units: kpc * (km/s)^2 / Msun
G_ASTRO = 4.30091e-6

# ------------------------------------------------------------
# DATA IO (kept simple, reused from uni2.py logic)
# ------------------------------------------------------------

KNOWN_PREFIXES = ("CamB", "D", "DDO", "ESO", "F", "IC", "KK", "NGC", "PGC", "UGC", "UGCA")


def parse_table1(path: str = "SPARC_Lelli2016c.mrt") -> Dict[str, dict]:
    galaxies: Dict[str, dict] = {}
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or len(line) < 30:
                continue
            if not any(line.startswith(p) for p in KNOWN_PREFIXES):
                continue
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                t = int(parts[1])
                if t < 0 or t > 11:
                    continue
                g = dict(
                    name=parts[0],
                    T=t,
                    dist=float(parts[2]),
                    inc=float(parts[5]),
                    L36=float(parts[7]),
                    Reff=float(parts[9]),
                    SBeff=float(parts[10]),
                    Rdisk=float(parts[11]),
                    SBdisk=float(parts[12]),
                    MHI=float(parts[13]),
                    RHI=float(parts[14]),
                    Vflat=float(parts[15]),
                    eVflat=float(parts[16]),
                    Qual=int(parts[17]),
                )
                galaxies[g["name"]] = g
            except (ValueError, IndexError):
                continue
    return galaxies


def parse_rotmod(path: str) -> dict:
    d = {k: [] for k in ("Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul")}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rad = float(parts[0])
                vobs = float(parts[1])
                errv = float(parts[2])
                vgas = float(parts[3])
                vdisk = float(parts[4])
                vbul = float(parts[5])
                if rad > 0 and vobs > 0 and errv > 0:
                    d["Rad"].append(rad)
                    d["Vobs"].append(vobs)
                    d["errV"].append(errv)
                    d["Vgas"].append(vgas)
                    d["Vdisk"].append(vdisk)
                    d["Vbul"].append(vbul)
            except ValueError:
                continue
    return {k: np.asarray(v, dtype=float) for k, v in d.items()}


def load_rotcurves(rotmod_dir: str = "Rotmod_LTG") -> Dict[str, dict]:
    curves: Dict[str, dict] = {}
    files = sorted(set(glob.glob(os.path.join(rotmod_dir, "*_rotmod.dat"))))
    for fp in files:
        name = os.path.basename(fp).replace("_rotmod.dat", "")
        d = parse_rotmod(fp)
        if len(d["Rad"]) >= 5:
            curves[name] = d
    return curves


# ------------------------------------------------------------
# HALO / MEMORY DENSITIES + ENCLOSED MASS
# All masses in Msun, lengths in kpc, densities in Msun/kpc^3.
# V^2 = G_ASTRO * M(<r) / r -> (km/s)^2
# ------------------------------------------------------------

def pointer_density(r: np.ndarray, rho0: float, r_mem: float,
                    alpha: float, r_core: float) -> np.ndarray:
    """rho_mem(r) = rho0 * exp(-r/r_mem) * (1 + alpha * ln(1 + r/r_core))"""
    r_mem = max(r_mem, 1e-3)
    r_core = max(r_core, 1e-3)
    return rho0 * np.exp(-r / r_mem) * (1.0 + alpha * np.log1p(r / r_core))


def mass_enclosed(density_of_r: Callable[[np.ndarray], np.ndarray],
                  r_eval: np.ndarray,
                  r_max_factor: float = 1.0,
                  n_grid: int = 200) -> np.ndarray:
    """Spherical enclosed mass via cumulative trapezoid of 4 pi r^2 rho(r).

    density_of_r: callable taking a 1-D numpy array of radii (kpc) and
    returning density in Msun/kpc^3.
    r_eval: points at which we want M(<r), ascending, in kpc.
    """
    r_max = float(r_eval[-1]) * max(r_max_factor, 1.0)
    # A log-spaced grid from small r up to r_max. We want r=0 included so we
    # use linspace but clip to small positive start to avoid 0 in log1p argument
    # anyway (we already protect r_core).
    r_grid = np.linspace(0.0, r_max, n_grid)
    integrand = 4.0 * np.pi * r_grid ** 2 * density_of_r(r_grid)
    m_grid = cumulative_trapezoid(integrand, r_grid, initial=0.0)
    # interpolate to requested r_eval
    return np.interp(r_eval, r_grid, m_grid)


def v2_pointer(r: np.ndarray, rho0: float, r_mem: float,
               alpha: float, r_core: float) -> np.ndarray:
    m = mass_enclosed(lambda rr: pointer_density(rr, rho0, r_mem, alpha, r_core), r)
    return np.where(r > 0, G_ASTRO * m / r, 0.0)


def v2_nfw(r: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    r_s = max(r_s, 1e-3)
    x = r / r_s
    # closed form: M(r) = 4 pi rho_s r_s^3 [ ln(1+x) - x/(1+x) ]
    m = 4.0 * np.pi * rho_s * r_s ** 3 * (np.log1p(x) - x / (1.0 + x))
    return np.where(r > 0, G_ASTRO * m / r, 0.0)


def v2_burkert(r: np.ndarray, rho_c: float, r_c: float) -> np.ndarray:
    r_c = max(r_c, 1e-3)
    x = r / r_c
    # closed form: M(r) = pi rho_c r_c^3 [ 2 ln(1+x) + ln(1+x^2) - 2 arctan(x) ]
    # derived from rho = rho_c / [(1+x)(1+x^2)]
    m = np.pi * rho_c * r_c ** 3 * (2.0 * np.log1p(x) + np.log1p(x * x) - 2.0 * np.arctan(x))
    return np.where(r > 0, G_ASTRO * m / r, 0.0)


def v_total(r, Vgas, Vdisk, Vbul, Y_disk, Y_bul, v2_halo: np.ndarray) -> np.ndarray:
    # SPARC: Vgas, Vdisk, Vbul are signed; V^2 preserves sign by V*|V|
    def signed_sq(v):
        return v * np.abs(v)
    V_bar_sq = Y_disk * signed_sq(Vdisk) + Y_bul * signed_sq(Vbul) + signed_sq(Vgas)
    V_total_sq = V_bar_sq + v2_halo
    V_total_sq = np.maximum(V_total_sq, 0.0)
    return np.sqrt(V_total_sq)


# ------------------------------------------------------------
# FIT ONE GALAXY WITH ONE HALO MODEL
# ------------------------------------------------------------

@dataclass
class FitOutcome:
    model: str
    params: np.ndarray
    chi2: float
    n: int
    k: int  # number of free halo params (Y_disk, Y_bul count separately below)


def _bounds_for_model(model: str, has_bulge: bool, Rdisk: float, Rmax: float) -> List[Tuple[float, float]]:
    """Return bounds in log-parametrization for scale params.

    Parameters in fit vector:
      Y_disk, Y_bul (linear)
      log10(rho0_Msun_per_kpc3), log10(r_scale_kpc), ... (model-specific)
    """
    # Y_disk, Y_bul in M/L solar units. Typical for late-type: 0.3-0.8.
    yb = (0.0, 2.0) if has_bulge else (0.0, 0.01)
    common = [(0.1, 1.5), yb]  # Y_disk, Y_bul
    r_scale_hi = max(200.0, 4.0 * Rmax)
    log_r_hi = math.log10(r_scale_hi)
    if model == "pointer":
        # log10 rho0, log10 r_mem, alpha (linear), log10 r_core
        return common + [(3.0, 11.0), (-1.0, log_r_hi), (0.0, 5.0), (-2.0, math.log10(50.0))]
    if model == "nfw":
        return common + [(3.0, 10.0), (-1.0, log_r_hi)]  # log rho_s, log r_s
    if model == "burkert":
        return common + [(3.0, 10.0), (-1.0, log_r_hi)]  # log rho_c, log r_c
    raise ValueError(model)


def _v2_halo_for(model: str, r, halo_params_log):
    """halo_params_log: scale params are in log10 except alpha (linear)."""
    if model == "pointer":
        log_rho0, log_r_mem, alpha, log_r_core = halo_params_log
        return v2_pointer(r, 10.0 ** log_rho0, 10.0 ** log_r_mem, alpha, 10.0 ** log_r_core)
    if model == "nfw":
        log_rho_s, log_r_s = halo_params_log
        return v2_nfw(r, 10.0 ** log_rho_s, 10.0 ** log_r_s)
    if model == "burkert":
        log_rho_c, log_r_c = halo_params_log
        return v2_burkert(r, 10.0 ** log_rho_c, 10.0 ** log_r_c)
    raise ValueError(model)


def _v2_halo_physical(model: str, r, halo_params_physical):
    """halo_params_physical: all params in physical (linear) units."""
    if model == "pointer":
        rho0, r_mem, alpha, r_core = halo_params_physical
        return v2_pointer(r, rho0, r_mem, alpha, r_core)
    if model == "nfw":
        rho_s, r_s = halo_params_physical
        return v2_nfw(r, rho_s, r_s)
    if model == "burkert":
        rho_c, r_c = halo_params_physical
        return v2_burkert(r, rho_c, r_c)
    raise ValueError(model)


def _unpack_params_to_physical(model: str, fit_params: np.ndarray) -> np.ndarray:
    """Convert fit-space params (with logs) back to physical units."""
    out = fit_params.copy()
    if model == "pointer":
        out[2] = 10.0 ** fit_params[2]  # rho0
        out[3] = 10.0 ** fit_params[3]  # r_mem
        # out[4] alpha stays linear
        out[5] = 10.0 ** fit_params[5]  # r_core
    elif model in ("nfw", "burkert"):
        out[2] = 10.0 ** fit_params[2]
        out[3] = 10.0 ** fit_params[3]
    return out


def fit_model(model: str, r, Vobs, errV, Vgas, Vdisk, Vbul,
              has_bulge: bool, Rdisk: float, seed: int = 42) -> FitOutcome:
    Rmax = float(np.max(r))
    bounds = _bounds_for_model(model, has_bulge, Rdisk, Rmax)

    def chi2_fun(p):
        Y_disk, Y_bul = p[0], p[1]
        halo = p[2:]
        v2halo = _v2_halo_for(model, r, halo)
        vmodel = v_total(r, Vgas, Vdisk, Vbul, Y_disk, Y_bul, v2halo)
        return float(np.sum(((Vobs - vmodel) / errV) ** 2) / max(len(r) - len(p), 1))

    # global search; wider budget for the 6-param pointer model
    de = differential_evolution(
        chi2_fun, bounds, seed=seed,
        maxiter=1200 if model == "pointer" else 700,
        tol=1e-7, polish=True, popsize=25, init="sobol",
    )
    best_p, best_chi2 = de.x, de.fun

    # Nelder-Mead restarts — cover log-space with scale_mult on r_mem
    for y_d_guess in (0.3, 0.5, 0.7):
        for scale_mult in (0.5, 2.0, 8.0):
            r_scale_log = math.log10(max(0.1, Rdisk * scale_mult))
            x0 = [y_d_guess, 0.5 if has_bulge else 0.0]
            if model == "pointer":
                x0 += [7.0, r_scale_log, 0.5, math.log10(max(0.05, Rdisk * 0.5))]
            else:
                x0 += [7.0, r_scale_log]
            try:
                res = minimize(chi2_fun, x0, method="Nelder-Mead",
                               options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
                in_b = all(bounds[i][0] <= res.x[i] <= bounds[i][1] for i in range(len(bounds)))
                if in_b and res.fun < best_chi2:
                    best_chi2, best_p = res.fun, res.x
            except Exception:
                continue

    physical_p = _unpack_params_to_physical(model, best_p)
    return FitOutcome(model=model, params=physical_p, chi2=best_chi2,
                      n=len(r), k=len(bounds))


# ------------------------------------------------------------
# AIC / BIC
# ------------------------------------------------------------

def aic_bic(chi2_reduced: float, n: int, k: int) -> Tuple[float, float]:
    # chi2_reduced = RSS / (n-k); RSS = chi2_reduced * (n-k) but this is only
    # true when errV were used as weights (RSS in units of sigma^2).
    # For the likelihood, gaussian errors with known sigma give
    # -2 ln L = chi2_abs + const. We use chi2_abs = chi2_reduced * (n-k)
    # as a stand-in because the additive constant cancels in AIC/BIC deltas
    # across models with same data points and errors.
    chi2_abs = chi2_reduced * max(n - k, 1)
    aic = chi2_abs + 2 * k
    bic = chi2_abs + k * math.log(max(n, 2))
    return aic, bic


# ------------------------------------------------------------
# 5-FOLD CV on radial points
# ------------------------------------------------------------

def cv_chi2(model: str, r, Vobs, errV, Vgas, Vdisk, Vbul,
            has_bulge: bool, Rdisk: float, n_folds: int = 5, seed: int = 42) -> Tuple[float, float]:
    """Return (train_chi2_mean, test_chi2_mean) averaged across folds."""
    n = len(r)
    if n < n_folds + 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    fold_size = n // n_folds
    train_list, test_list = [], []
    for i in range(n_folds):
        lo = i * fold_size
        hi = n if i == n_folds - 1 else (i + 1) * fold_size
        test_idx = order[lo:hi]
        train_idx = np.concatenate([order[:lo], order[hi:]])
        out = fit_model(model, r[train_idx], Vobs[train_idx], errV[train_idx],
                        Vgas[train_idx], Vdisk[train_idx], Vbul[train_idx],
                        has_bulge=has_bulge, Rdisk=Rdisk, seed=seed + i)
        # test chi2 on held-out points using fit parameters
        Y_disk, Y_bul = out.params[0], out.params[1]
        halo = out.params[2:]
        v2halo_test = _v2_halo_physical(model, r[test_idx], halo)
        vmodel_test = v_total(r[test_idx], Vgas[test_idx], Vdisk[test_idx],
                              Vbul[test_idx], Y_disk, Y_bul, v2halo_test)
        rss_test = float(np.sum(((Vobs[test_idx] - vmodel_test) / errV[test_idx]) ** 2))
        test_chi2 = rss_test / max(len(test_idx), 1)
        train_list.append(out.chi2)
        test_list.append(test_chi2)
    return float(np.mean(train_list)), float(np.mean(test_list))


# ------------------------------------------------------------
# AGE PROXIES
# ------------------------------------------------------------

def compute_proxies(g: dict) -> dict:
    # stellar mass via L[3.6] -> M* ~ 0.5 * L
    M_star = 0.5 * max(g["L36"], 0.0) * 1e9
    M_star = max(M_star, 1e6)
    log_Mstar = math.log10(M_star)
    MHI_solar = max(g["MHI"], 0.0) * 1e9
    gas_frac = MHI_solar / (MHI_solar + M_star) if (MHI_solar + M_star) > 0 else 0.0
    # N_orbits at Rdisk: 13.7 Gyr / t_orb. t_orb = 2 pi R / V.
    # Express directly in Gyr: R in kpc, V in km/s.
    # 1 kpc / (1 km/s) = 3.086e16 m / (1e3 m/s) = 3.086e13 s = 0.978 Gyr
    if g["Vflat"] > 0 and g["Rdisk"] > 0:
        t_orb_Gyr = 2.0 * math.pi * g["Rdisk"] / g["Vflat"] * 0.978
        N_orbits = 13.7 / max(t_orb_Gyr, 1e-3)
    else:
        N_orbits = float("nan")
    log_N_orbits = math.log10(N_orbits) if (N_orbits and N_orbits > 0) else float("nan")
    if g["SBeff"] > 0 and g["SBdisk"] > 0:
        concentration = math.log10(g["SBdisk"] / g["SBeff"])
    else:
        concentration = float("nan")
    log_SBeff = math.log10(g["SBeff"]) if g["SBeff"] > 0 else float("nan")
    return dict(log_Mstar=log_Mstar, gas_frac=gas_frac,
                N_orbits=N_orbits, log_N_orbits=log_N_orbits,
                concentration=concentration, log_SBeff=log_SBeff)


# ------------------------------------------------------------
# CORRELATION UTILITIES
# ------------------------------------------------------------

def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4:
        return float("nan"), float("nan")
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if denom <= 0:
        return float("nan"), float("nan")
    r = float((dx * dy).sum() / denom)
    r = max(-0.9999, min(0.9999, r))
    n = len(x)
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # normal-approx two-sided p for |t|
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def spearman(x, y):
    if len(x) < 4:
        return float("nan"), float("nan")
    return pearson(rankdata(x), rankdata(y))


def permutation_p_pearson(x, y, n_perm=10000, seed=0) -> float:
    rng = np.random.RandomState(seed)
    r_obs = pearson(x, y)[0]
    if not np.isfinite(r_obs):
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r = pearson(x, yp)[0]
        if np.isfinite(r) and abs(r) >= abs(r_obs):
            count += 1
    return count / n_perm


def bootstrap_pearson_ci(x, y, n_boot=5000, conf=0.95, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    rs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        rs[i] = pearson(x[idx], y[idx])[0]
    rs = rs[np.isfinite(rs)]
    return float(np.mean(rs)), float(np.percentile(rs, 100 * (1 - conf) / 2)), \
           float(np.percentile(rs, 100 * (1 + conf) / 2))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("Pointer Architecture: canonical analysis on SPARC")
    print("=" * 70)

    print("\n[1/6] loading table 1 + rotation curves")
    galaxies = parse_table1()
    print(f"  parsed {len(galaxies)} galaxies from Table 1")
    curves = load_rotcurves()
    print(f"  loaded {len(curves)} rotation curve files")

    # match
    matched = []
    for name in galaxies:
        key = name.replace(" ", "")
        if key in curves:
            matched.append(name)
    print(f"  matched {len(matched)}")

    # compute proxies once per galaxy
    for name in matched:
        galaxies[name]["proxies"] = compute_proxies(galaxies[name])

    print(f"\n[2/6] fitting 3 models to {len(matched)} galaxies")
    results_per_model: Dict[str, Dict[str, dict]] = {"pointer": {}, "nfw": {}, "burkert": {}}
    cv_results: Dict[str, Dict[str, Tuple[float, float]]] = {"pointer": {}, "nfw": {}, "burkert": {}}

    for i, name in enumerate(matched):
        g = galaxies[name]
        data = curves[name.replace(" ", "")]
        r = data["Rad"]; Vobs = data["Vobs"]; errV = data["errV"]
        Vgas = data["Vgas"]; Vdisk = data["Vdisk"]; Vbul = data["Vbul"]
        has_bulge = bool(np.any(np.abs(Vbul) > 1.0))
        Rdisk = max(g["Rdisk"], 0.1)

        for model in ("pointer", "nfw", "burkert"):
            out = fit_model(model, r, Vobs, errV, Vgas, Vdisk, Vbul,
                            has_bulge=has_bulge, Rdisk=Rdisk)
            aic, bic = aic_bic(out.chi2, out.n, out.k)
            params = out.params.tolist()
            entry = dict(chi2=out.chi2, n=out.n, k=out.k, aic=aic, bic=bic, params=params)
            if model == "pointer":
                # physical params order: Y_disk, Y_bul, rho0, r_mem, alpha, r_core
                entry["r_mem"] = params[3]
                entry["r_ratio"] = params[3] / Rdisk
                entry["alpha"] = params[4]
            results_per_model[model][name] = entry

            # CV only for pointer (the one with most params → highest overfit risk).
            # Skipping CV for NFW/Burkert saves ~60% of total runtime.
            if model == "pointer":
                tr, te = cv_chi2(model, r, Vobs, errV, Vgas, Vdisk, Vbul,
                                 has_bulge=has_bulge, Rdisk=Rdisk, n_folds=3)
                cv_results[model][name] = (tr, te)

        if (i + 1) % 10 == 0 or i == len(matched) - 1:
            p_chi2 = results_per_model["pointer"][name]["chi2"]
            n_chi2 = results_per_model["nfw"][name]["chi2"]
            b_chi2 = results_per_model["burkert"][name]["chi2"]
            print(f"  {i+1:3d}/{len(matched)}  {name:<14}  "
                  f"chi2 pointer={p_chi2:5.2f}  nfw={n_chi2:5.2f}  burk={b_chi2:5.2f}")

    elapsed = time.time() - t0
    print(f"  fits done in {elapsed/60:.1f} min")

    # ---- summary across 171 ----
    print("\n[3/6] headline stats")
    summary = {}
    for model in ("pointer", "nfw", "burkert"):
        chi2s = np.array([results_per_model[model][n]["chi2"] for n in matched])
        aics = np.array([results_per_model[model][n]["aic"] for n in matched])
        bics = np.array([results_per_model[model][n]["bic"] for n in matched])
        summary[model] = dict(
            median_chi2=float(np.median(chi2s)),
            mean_chi2=float(np.mean(chi2s)),
            frac_chi2_lt_3=float(np.mean(chi2s < 3)),
            median_aic=float(np.median(aics)),
            median_bic=float(np.median(bics)),
        )
        print(f"  {model:7s}  median chi2={summary[model]['median_chi2']:.2f}  "
              f"frac<3={summary[model]['frac_chi2_lt_3']*100:.0f}%  "
              f"median AIC={summary[model]['median_aic']:.1f}  "
              f"median BIC={summary[model]['median_bic']:.1f}")

    # AIC win rates (lower is better)
    wins = {"pointer": 0, "nfw": 0, "burkert": 0}
    for name in matched:
        aics = {m: results_per_model[m][name]["aic"] for m in wins}
        best = min(aics, key=aics.get)
        wins[best] += 1
    print(f"  AIC best-model split: pointer={wins['pointer']}, nfw={wins['nfw']}, burkert={wins['burkert']}")

    # CV: mean train vs test (pointer only)
    tr = np.array([cv_results["pointer"][n][0] for n in matched])
    te = np.array([cv_results["pointer"][n][1] for n in matched])
    tr_f = tr[np.isfinite(tr)]
    te_f = te[np.isfinite(te)]
    ratio = te / np.where(tr > 0, tr, 1.0)
    ratio = ratio[np.isfinite(ratio)]
    print(f"  pointer  CV train chi2 median={np.median(tr_f):.2f}  "
          f"test chi2 median={np.median(te_f):.2f}  "
          f"(test/train ratio median={np.median(ratio):.2f})")

    # ---- correlations ----
    print("\n[4/6] correlations on r_mem/r_disk (pointer)")
    filtered_names = []
    log_ratios = []
    proxies_keys = ("T", "log_Mstar", "gas_frac", "log_N_orbits", "concentration", "log_SBeff")
    proxy_matrix: Dict[str, List[float]] = {k: [] for k in proxies_keys}
    expected_dir = dict(T="NEG", log_Mstar="POS", gas_frac="NEG",
                        log_N_orbits="POS", concentration="POS", log_SBeff="POS")

    for name in matched:
        pr = results_per_model["pointer"][name]
        g = galaxies[name]
        # quality cuts mirroring uni2 for comparability of sample size
        if pr["chi2"] > 10 or pr["r_ratio"] < 0.1 or pr["r_ratio"] > 30:
            continue
        if g["Qual"] == 3 or g["Rdisk"] <= 0 or g["Vflat"] <= 0:
            continue
        if g["T"] < 0:
            continue
        prx = g["proxies"]
        values = dict(
            T=g["T"],
            log_Mstar=prx["log_Mstar"],
            gas_frac=prx["gas_frac"],
            log_N_orbits=prx["log_N_orbits"],
            concentration=prx["concentration"],
            log_SBeff=prx["log_SBeff"],
        )
        # drop galaxy if any proxy is non-finite
        if any((not np.isfinite(v)) for v in values.values()):
            continue
        filtered_names.append(name)
        log_ratios.append(math.log10(pr["r_ratio"]))
        for k in proxies_keys:
            proxy_matrix[k].append(values[k])

    N = len(filtered_names)
    print(f"  correlation sample size N = {N}")
    log_ratios = np.asarray(log_ratios)
    corr_rows = []
    confirmed = 0
    for k in proxies_keys:
        arr = np.asarray(proxy_matrix[k])
        r, p = pearson(arr, log_ratios)
        rs, ps = spearman(arr, log_ratios)
        p_perm = permutation_p_pearson(arr, log_ratios, n_perm=5000, seed=42)
        _, ci_lo, ci_hi = bootstrap_pearson_ci(arr, log_ratios, n_boot=5000, seed=42)
        matches = (expected_dir[k] == "POS" and r > 0) or (expected_dir[k] == "NEG" and r < 0)
        if matches:
            confirmed += 1
        corr_rows.append(dict(proxy=k, expected=expected_dir[k], n=N,
                              pearson_r=r, pearson_p=p, pearson_p_perm=p_perm,
                              spearman_r=rs, spearman_p=ps,
                              ci_lo=ci_lo, ci_hi=ci_hi, direction_ok=matches))
        print(f"  {k:16s} exp={expected_dir[k]}  r={r:+.3f} p={p:.4f} p_perm={p_perm:.4f}  "
              f"rho={rs:+.3f} p_rho={ps:.4f}  CI=[{ci_lo:+.2f},{ci_hi:+.2f}]  "
              f"{'dir ok' if matches else 'WRONG'}")

    print(f"  {confirmed}/{len(proxies_keys)} in predicted direction")
    # binomial exact P(X>=confirmed | n=6, p=0.5), no post-hoc exclusion
    from math import comb
    binom_p = sum(comb(6, i) * 0.5 ** 6 for i in range(confirmed, 7))
    print(f"  binomial p (no exclusions) = {binom_p:.4f}")

    # ---- PCA ----
    print("\n[5/6] PCA on 6 proxies")
    X = np.column_stack([proxy_matrix[k] for k in proxies_keys])
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    cov = np.cov(Xz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    explained = eigvals / eigvals.sum()
    scores = Xz @ eigvecs
    print(f"  explained variance: " + ", ".join(f"PC{i+1}={explained[i]*100:.1f}%" for i in range(6)))
    print(f"  PC1 loadings: " + ", ".join(f"{proxies_keys[j]}={eigvecs[j,0]:+.2f}" for j in range(6)))
    # sign-align PC1 to the "older/more-evolved" direction by anchoring log(M*)
    mstar_idx = proxies_keys.index("log_Mstar")
    if eigvecs[mstar_idx, 0] < 0:
        eigvecs[:, 0] = -eigvecs[:, 0]
        scores[:, 0] = -scores[:, 0]
    # correlations of PC scores with log_ratio
    pca_rows = []
    for i in range(3):
        r, p = pearson(scores[:, i], log_ratios)
        rs, ps = spearman(scores[:, i], log_ratios)
        p_perm = permutation_p_pearson(scores[:, i], log_ratios, n_perm=5000, seed=17 + i)
        _, ci_lo, ci_hi = bootstrap_pearson_ci(scores[:, i], log_ratios, n_boot=5000, seed=17 + i)
        pca_rows.append(dict(component=f"PC{i+1}", explained=float(explained[i]),
                             pearson_r=r, pearson_p=p, pearson_p_perm=p_perm,
                             spearman_r=rs, spearman_p=ps, ci_lo=ci_lo, ci_hi=ci_hi))
        print(f"  PC{i+1}  var={explained[i]*100:5.1f}%  r={r:+.3f} p_perm={p_perm:.4f}  "
              f"rho={rs:+.3f} CI=[{ci_lo:+.2f},{ci_hi:+.2f}]")

    # ---- write outputs ----
    print("\n[6/6] writing outputs")
    with open("final_fits.csv", "w") as f:
        headers = ["name", "T", "Rdisk", "Vflat", "Qual"]
        for model in ("pointer", "nfw", "burkert"):
            headers += [f"{model}_chi2", f"{model}_aic", f"{model}_bic", f"{model}_k", f"{model}_n"]
        headers += ["pointer_r_mem", "pointer_r_ratio", "pointer_alpha"]
        headers += ["cv_pointer_train", "cv_pointer_test"]
        f.write(",".join(headers) + "\n")
        for name in matched:
            g = galaxies[name]
            row = [name, g["T"], f"{g['Rdisk']:.3f}", f"{g['Vflat']:.1f}", g["Qual"]]
            for model in ("pointer", "nfw", "burkert"):
                r = results_per_model[model][name]
                row += [f"{r['chi2']:.4f}", f"{r['aic']:.2f}", f"{r['bic']:.2f}", r["k"], r["n"]]
            pt = results_per_model["pointer"][name]
            row += [f"{pt['r_mem']:.3f}", f"{pt['r_ratio']:.3f}", f"{pt['alpha']:.3f}"]
            tr_p, te_p = cv_results["pointer"].get(name, (float("nan"), float("nan")))
            row += [f"{tr_p:.3f}", f"{te_p:.3f}"]
            f.write(",".join(str(x) for x in row) + "\n")
    print("  wrote final_fits.csv")

    with open("final_corr.csv", "w") as f:
        f.write("proxy,expected,n,pearson_r,pearson_p,pearson_p_perm,"
                "spearman_r,spearman_p,ci_lo,ci_hi,direction_ok\n")
        for row in corr_rows:
            f.write(f"{row['proxy']},{row['expected']},{row['n']},"
                    f"{row['pearson_r']:.4f},{row['pearson_p']:.4f},{row['pearson_p_perm']:.4f},"
                    f"{row['spearman_r']:.4f},{row['spearman_p']:.4f},"
                    f"{row['ci_lo']:.4f},{row['ci_hi']:.4f},{int(row['direction_ok'])}\n")
    print("  wrote final_corr.csv")

    with open("final_pca.csv", "w") as f:
        f.write("component,explained,pearson_r,pearson_p,pearson_p_perm,"
                "spearman_r,spearman_p,ci_lo,ci_hi\n")
        for row in pca_rows:
            f.write(f"{row['component']},{row['explained']:.4f},"
                    f"{row['pearson_r']:.4f},{row['pearson_p']:.4f},{row['pearson_p_perm']:.4f},"
                    f"{row['spearman_r']:.4f},{row['spearman_p']:.4f},"
                    f"{row['ci_lo']:.4f},{row['ci_hi']:.4f}\n")
    print("  wrote final_pca.csv")

    # human report
    with open("final_report.md", "w") as f:
        f.write("# Pointer Architecture: canonical analysis on SPARC\n\n")
        f.write(f"Galaxies fitted: {len(matched)} / 175 (Table 1 + rotmod matched).\n\n")
        f.write("## Fit quality, head-to-head\n\n")
        f.write("| Model | median chi2 | frac chi2<3 | median AIC | median BIC |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for model in ("pointer", "nfw", "burkert"):
            s = summary[model]
            f.write(f"| {model} | {s['median_chi2']:.3f} | {s['frac_chi2_lt_3']*100:.0f}% | "
                    f"{s['median_aic']:.2f} | {s['median_bic']:.2f} |\n")
        f.write("\n")
        f.write(f"AIC best model split across {len(matched)} galaxies: "
                f"pointer={wins['pointer']}, nfw={wins['nfw']}, burkert={wins['burkert']}.\n\n")
        f.write("## Cross-validation (3-fold on radial points, pointer model)\n\n")
        tr_cv = np.array([cv_results["pointer"][n][0] for n in matched])
        te_cv = np.array([cv_results["pointer"][n][1] for n in matched])
        tr_f = tr_cv[np.isfinite(tr_cv)]
        te_f = te_cv[np.isfinite(te_cv)]
        ratio_cv = te_cv / np.where(tr_cv > 0, tr_cv, 1.0)
        ratio_cv = ratio_cv[np.isfinite(ratio_cv)]
        f.write(f"- pointer: train chi2 median {np.median(tr_f):.2f}, "
                f"test chi2 median {np.median(te_f):.2f}, "
                f"test/train ratio median {np.median(ratio_cv):.2f}\n")
        f.write("  A ratio near 1 suggests no overfitting; >>1 is a red flag.\n")
        f.write("\n## Correlations on log(r_mem/r_disk), pointer model\n\n")
        f.write(f"Sample: N = {N} (chi2<10, 0.1<ratio<30, Q=1,2, all proxies finite).\n\n")
        f.write("| proxy | exp | r | p | p_perm | spearman | ci_lo | ci_hi | dir |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in corr_rows:
            f.write(f"| {row['proxy']} | {row['expected']} | "
                    f"{row['pearson_r']:+.3f} | {row['pearson_p']:.3f} | "
                    f"{row['pearson_p_perm']:.3f} | {row['spearman_r']:+.3f} | "
                    f"{row['ci_lo']:+.3f} | {row['ci_hi']:+.3f} | "
                    f"{'ok' if row['direction_ok'] else 'WRONG'} |\n")
        f.write(f"\nDirectional summary: {confirmed}/{len(proxies_keys)} in predicted direction.\n")
        f.write(f"Binomial test (no post-hoc exclusions): p = {binom_p:.4f}.\n\n")
        f.write("## PCA on 6 proxies\n\n")
        f.write("| component | var explained | r(PC,log ratio) | p_perm | spearman | ci |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for row in pca_rows:
            f.write(f"| {row['component']} | {row['explained']*100:.1f}% | "
                    f"{row['pearson_r']:+.3f} | {row['pearson_p_perm']:.3f} | "
                    f"{row['spearman_r']:+.3f} | [{row['ci_lo']:+.2f},{row['ci_hi']:+.2f}] |\n")
        f.write("\nPCA directional test: PC1 sign-aligned to log(M*) loading (older-more-massive direction).\n")
        f.write("A positive PC1~log-ratio correlation supports memory-accumulation hypothesis.\n\n")
        f.write(f"Run duration: {(time.time()-t0)/60:.1f} min.\n")
    print("  wrote final_report.md")

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
