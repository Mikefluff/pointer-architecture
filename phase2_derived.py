#!/usr/bin/env python3
"""
Phase 2 — Derived observables on top of the canonical analysis.

Rationale:
  The canonical analysis tests the ratio r_mem/r_disk against age proxies.
  But the memory-accumulation hypothesis predicts that accumulated *mass*
  grows with computational history, not that a scale length grows.
  This module adds the physically-direct observables:
    - M_halo(<R_last) / M_bar  (archive-to-baryon mass ratio)
    - f_DM(R_last) = V^2_halo / V^2_obs at the last observed radius
  Both are computed from the pointer fits in final_fits.csv.

  Proxies are cleaned: N_orbits and concentration are dropped (N_orbits is
  a rediscovery of V/R that is already in the fit; concentration has no
  age signal in SPARC — r = -0.002 in the canonical run).
  A single composite "maturity" score is built as PC1 of the 4 remaining
  proxies (log_M*, T-type, gas_frac, log SBeff), with weights derived on
  the full 171-galaxy sample so that the sub-sample filtering does not
  leak into the direction definition.

Inputs:
  - SPARC_Lelli2016c.mrt
  - Rotmod_LTG/*_rotmod.dat
  - final_fits.csv    (output of methodology_final.py)

Outputs:
  - phase2_derived.csv
  - phase2_report.md
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import rankdata, wilcoxon

import analysis as base  # reuse parse_table1, parse_rotmod, densities

G_ASTRO = base.G_ASTRO


# ------------------------------------------------------------
# DERIVED OBSERVABLES
# ------------------------------------------------------------

def pointer_mass_enclosed(r_last: float, rho0: float, r_mem: float,
                          alpha: float, r_core: float,
                          n_grid: int = 400) -> float:
    """M_halo(<r_last) for the pointer density profile, in Msun."""
    grid = np.linspace(0.0, r_last, n_grid)
    integrand = 4.0 * np.pi * grid ** 2 * base.pointer_density(
        grid, rho0, r_mem, alpha, r_core)
    m_grid = cumulative_trapezoid(integrand, grid, initial=0.0)
    return float(m_grid[-1])


def baryonic_velocity_sq_at(r_last: float, y_disk: float, y_bul: float,
                            rotmod: dict) -> float:
    """V^2_bar at r_last, evaluated by interpolating the SPARC rotmod."""
    r = rotmod["Rad"]
    Vgas = np.interp(r_last, r, rotmod["Vgas"])
    Vdisk = np.interp(r_last, r, rotmod["Vdisk"])
    Vbul = np.interp(r_last, r, rotmod["Vbul"])

    def signed_sq(v):
        return v * abs(v)

    return (y_disk * signed_sq(Vdisk)
            + y_bul * signed_sq(Vbul)
            + signed_sq(Vgas))


def baryonic_mass_Msun(g: dict) -> float:
    """M_bar = M* + 1.33 * M_HI. Both from SPARC Table 1, converted to Msun."""
    M_star = 0.5 * max(g["L36"], 0.0) * 1e9          # 0.5 Msun/Lsun at 3.6 um
    M_gas = 1.33 * max(g["MHI"], 0.0) * 1e9          # He correction
    return max(M_star + M_gas, 1e6)


# ------------------------------------------------------------
# CORRELATION UTILITIES (independent of methodology_final)
# ------------------------------------------------------------

def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    rx = rankdata(x); ry = rankdata(y)
    return pearson(rx, ry)


def pearson(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 4:
        return float("nan"), float("nan")
    dx = x - x.mean(); dy = y - y.mean()
    denom = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if denom <= 0:
        return float("nan"), float("nan")
    r = float((dx * dy).sum() / denom)
    r = max(-0.9999, min(0.9999, r))
    n = len(x)
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def permutation_p_spearman(x, y, n_perm=10000, seed=0):
    rng = np.random.RandomState(seed)
    r_obs = spearman(x, y)[0]
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if not np.isfinite(r_obs):
        return float("nan")
    count = 0
    rx = rankdata(x)
    ry = rankdata(y)
    for _ in range(n_perm):
        ryp = rng.permutation(ry)
        r = pearson(rx, ryp)[0]
        if np.isfinite(r) and abs(r) >= abs(r_obs):
            count += 1
    return count / n_perm


def bootstrap_spearman_ci(x, y, n_boot=5000, conf=0.95, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    rs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        rs[i] = spearman(x[idx], y[idx])[0]
    rs = rs[np.isfinite(rs)]
    return float(np.mean(rs)), \
           float(np.percentile(rs, 100 * (1 - conf) / 2)), \
           float(np.percentile(rs, 100 * (1 + conf) / 2))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def load_fits_csv(path: str = "final_fits.csv") -> Dict[str, dict]:
    res: Dict[str, dict] = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["name"]
            # pointer params were serialised via list(); r_mem, r_ratio, alpha
            # saved as named columns.
            res[name] = dict(
                T=int(row["T"]),
                Rdisk=float(row["Rdisk"]),
                Vflat=float(row["Vflat"]),
                Qual=int(row["Qual"]),
                pointer_chi2=float(row["pointer_chi2"]),
                pointer_aic=float(row["pointer_aic"]),
                nfw_aic=float(row["nfw_aic"]),
                burkert_aic=float(row["burkert_aic"]),
                pointer_r_mem=float(row["pointer_r_mem"]),
                pointer_r_ratio=float(row["pointer_r_ratio"]),
                pointer_alpha=float(row["pointer_alpha"]),
            )
    return res


def reconstruct_pointer_params(name: str, fits_row: dict,
                               curves: dict, galaxies: dict) -> dict:
    """We didn't serialise full pointer params in final_fits.csv.
    Refit quickly using the same pipeline to recover rho0, r_core.
    The per-galaxy fit is fast enough (~2s) because we already know the
    optimum is stable; we just need to re-run to extract rho0 and r_core.
    """
    g = galaxies[name]
    d = curves[name.replace(" ", "")]
    has_bulge = bool(np.any(np.abs(d["Vbul"]) > 1.0))
    out = base.fit_model("pointer", d["Rad"], d["Vobs"], d["errV"],
                         d["Vgas"], d["Vdisk"], d["Vbul"],
                         has_bulge=has_bulge, Rdisk=max(g["Rdisk"], 0.1))
    # out.params physical order: Y_disk, Y_bul, rho0, r_mem, alpha, r_core
    return dict(Y_disk=float(out.params[0]), Y_bul=float(out.params[1]),
                rho0=float(out.params[2]), r_mem=float(out.params[3]),
                alpha=float(out.params[4]), r_core=float(out.params[5]),
                chi2=float(out.chi2))


def main():
    print("Phase 2 — derived observables + composite age + Wilcoxon on alpha\n")

    galaxies = base.parse_table1()
    curves = base.load_rotcurves()

    # proxies on ALL 171 galaxies with rotation curves, for PC1 weights
    proxy_names_clean = ("log_Mstar", "T", "gas_frac", "log_SBeff")
    rows_all: List[dict] = []
    matched = []
    for name in galaxies:
        if name.replace(" ", "") in curves:
            g = galaxies[name]
            matched.append(name)
            prx = base.compute_proxies(g)
            T_val = g["T"]
            log_SB = prx["log_SBeff"]
            log_M = prx["log_Mstar"]
            gf = prx["gas_frac"]
            if not all(np.isfinite([T_val, log_SB, log_M, gf])):
                continue
            rows_all.append(dict(name=name, log_Mstar=log_M, T=T_val,
                                 gas_frac=gf, log_SBeff=log_SB))
    print(f"  171-galaxy proxy table has {len(rows_all)} usable rows")

    X = np.array([[r[k] for k in proxy_names_clean] for r in rows_all])
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    cov = np.cov(Xz, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]; eig_vecs = eig_vecs[:, idx]
    explained = eig_vals / eig_vals.sum()
    # sign-align so log_Mstar has a positive loading
    mstar_idx = proxy_names_clean.index("log_Mstar")
    if eig_vecs[mstar_idx, 0] < 0:
        eig_vecs[:, 0] = -eig_vecs[:, 0]
    pc1_weights = eig_vecs[:, 0]
    print(f"  PC1 explained {explained[0]*100:.1f}% of variance, "
          f"PC2 {explained[1]*100:.1f}%")
    print(f"  PC1 weights: " +
          ", ".join(f"{proxy_names_clean[i]}={pc1_weights[i]:+.3f}"
                    for i in range(4)))

    # name -> composite score (using the standardisation computed above)
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1)
    composite_age: Dict[str, float] = {}
    for r in rows_all:
        z = np.array([(r[k] - mu[i]) / sd[i] for i, k in enumerate(proxy_names_clean)])
        composite_age[r["name"]] = float(z @ pc1_weights)

    # load fits, reconstruct full pointer params, compute derived observables
    fits = load_fits_csv()
    print(f"\n  loaded {len(fits)} rows from final_fits.csv")

    derived_rows: List[dict] = []
    alpha_dist: List[float] = []
    for i, name in enumerate(matched):
        if name not in fits:
            continue
        g = galaxies[name]
        d = curves[name.replace(" ", "")]
        # Refit to recover full pointer params. Cached by seed so the result
        # is identical to the canonical run.
        p = reconstruct_pointer_params(name, fits[name], curves, galaxies)

        r_last = float(np.max(d["Rad"]))
        V_obs_last = float(np.interp(r_last, d["Rad"], d["Vobs"]))
        V_bar_sq = baryonic_velocity_sq_at(
            r_last, p["Y_disk"], p["Y_bul"], d)
        V_halo_sq = max(V_obs_last ** 2 - V_bar_sq, 0.0)
        f_DM = V_halo_sq / max(V_obs_last ** 2, 1.0)

        M_halo = pointer_mass_enclosed(
            r_last, p["rho0"], p["r_mem"], p["alpha"], p["r_core"])
        M_bar = baryonic_mass_Msun(g)
        M_ratio = M_halo / max(M_bar, 1.0)

        row = dict(
            name=name,
            T=g["T"], Rdisk=g["Rdisk"], Vflat=g["Vflat"], Qual=g["Qual"],
            pointer_chi2=p["chi2"],
            pointer_alpha=p["alpha"],
            pointer_rho0=p["rho0"],
            pointer_r_mem=p["r_mem"],
            pointer_r_core=p["r_core"],
            r_last=r_last,
            V_obs_last=V_obs_last,
            f_DM_Rlast=f_DM,
            M_halo_Msun=M_halo,
            M_bar_Msun=M_bar,
            M_ratio=M_ratio,
            composite_age=composite_age.get(name, float("nan")),
            r_ratio=fits[name]["pointer_r_ratio"],
        )
        derived_rows.append(row)
        alpha_dist.append(p["alpha"])
        if (i + 1) % 20 == 0 or i == len(matched) - 1:
            print(f"  {i+1:3d}/{len(matched)}  {name:<14}  "
                  f"f_DM={f_DM:.2f}  M_halo/M_bar={M_ratio:.2f}  "
                  f"alpha={p['alpha']:.2f}")

    # write derived CSV
    keys = ["name", "T", "Rdisk", "Vflat", "Qual",
            "pointer_chi2", "pointer_alpha", "pointer_rho0",
            "pointer_r_mem", "pointer_r_core",
            "r_last", "V_obs_last", "f_DM_Rlast",
            "M_halo_Msun", "M_bar_Msun", "M_ratio",
            "composite_age", "r_ratio"]
    with open("phase2_derived.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in derived_rows:
            vals = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    if abs(v) > 1e5 or (0 < abs(v) < 1e-3):
                        vals.append(f"{v:.6e}")
                    else:
                        vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            fh.write(",".join(vals) + "\n")
    print(f"\n  wrote phase2_derived.csv ({len(derived_rows)} rows)")

    # ---- Wilcoxon on alpha > 0 ----
    print("\n[Wilcoxon] population-level test on pointer alpha")
    alpha = np.array(alpha_dist)
    print(f"  n={len(alpha)}  median={np.median(alpha):.3f}  "
          f"frac>0.1={np.mean(alpha>0.1)*100:.0f}%  "
          f"frac>1={np.mean(alpha>1)*100:.0f}%  "
          f"frac at upper bound (>=4.9)={np.mean(alpha>=4.9)*100:.0f}%")
    # one-sided: median > 0
    try:
        w_stat, w_p = wilcoxon(alpha, alternative="greater")
        print(f"  Wilcoxon (alpha>0, one-sided): W={w_stat:.1f}, p={w_p:.2e}")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        print(f"  Wilcoxon failed: {e}")

    # ---- primary correlations: composite_age vs direct observables ----
    # Quality filters same as canonical: chi2<10, 0.1<r_ratio<30, Q in {1,2},
    # composite finite, observables finite.
    print("\n[primary] composite_age vs direct observables")
    good = []
    for row in derived_rows:
        if row["pointer_chi2"] > 10: continue
        if row["r_ratio"] < 0.1 or row["r_ratio"] > 30: continue
        if row["Qual"] == 3: continue
        if not np.isfinite(row["composite_age"]): continue
        if not np.isfinite(row["f_DM_Rlast"]): continue
        if row["M_ratio"] <= 0: continue
        good.append(row)
    N = len(good)
    print(f"  sample N = {N}")

    age = np.array([r["composite_age"] for r in good])
    observables = {
        "log r_ratio": np.log10(np.array([r["r_ratio"] for r in good])),
        "log M_halo/M_bar": np.log10(np.array([r["M_ratio"] for r in good])),
        "f_DM(R_last)": np.array([r["f_DM_Rlast"] for r in good]),
    }
    primary_rows = []
    for label, y in observables.items():
        r_s, p_s = spearman(age, y)
        r_p, p_p = pearson(age, y)
        p_perm = permutation_p_spearman(age, y, n_perm=10000, seed=99)
        _, ci_lo, ci_hi = bootstrap_spearman_ci(age, y, n_boot=5000, seed=99)
        primary_rows.append(dict(obs=label, n=N,
                                 spearman_r=r_s, spearman_p=p_s,
                                 pearson_r=r_p, pearson_p=p_p,
                                 perm_p=p_perm, ci_lo=ci_lo, ci_hi=ci_hi))
        print(f"  {label:<22}  rho={r_s:+.3f}  p_perm={p_perm:.4f}  "
              f"CI=[{ci_lo:+.3f},{ci_hi:+.3f}]  pearson r={r_p:+.3f} p={p_p:.4f}")

    # ---- report ----
    with open("phase2_report.md", "w") as fh:
        fh.write("# Phase 2 — direct observables and composite age\n\n")
        fh.write("## Construction\n\n")
        fh.write("Composite maturity score built as PC1 of 4 SPARC proxies "
                 "on the full 171-galaxy sample.\n")
        fh.write(f"PC1 explains {explained[0]*100:.1f}% of proxy variance "
                 f"(PC2 {explained[1]*100:.1f}%, PC3 {explained[2]*100:.1f}%).\n")
        fh.write("Weights:\n\n")
        for i, k in enumerate(proxy_names_clean):
            fh.write(f"- {k}: {pc1_weights[i]:+.3f}\n")
        fh.write("\nSign-aligned to log_Mstar = positive, so high composite_age "
                 "= more evolved galaxy.\n")
        fh.write("N_orbits and concentration are dropped from the proxy set: "
                 "the first duplicates the V/R information already inside the "
                 "fit; the second has r = -0.002 with log(r_ratio) and carries "
                 "no age signal.\n\n")

        fh.write("## Derived observables, 171-galaxy distribution\n\n")
        m_ratio = np.array([r["M_ratio"] for r in derived_rows if np.isfinite(r["M_ratio"])])
        f_dm = np.array([r["f_DM_Rlast"] for r in derived_rows if np.isfinite(r["f_DM_Rlast"])])
        fh.write(f"- M_halo/M_bar: median {np.median(m_ratio):.2f}, "
                 f"q25 {np.percentile(m_ratio,25):.2f}, "
                 f"q75 {np.percentile(m_ratio,75):.2f}\n")
        fh.write(f"- f_DM(R_last): median {np.median(f_dm):.2f}, "
                 f"q25 {np.percentile(f_dm,25):.2f}, "
                 f"q75 {np.percentile(f_dm,75):.2f}\n\n")

        fh.write("## Primary correlations: composite_age vs direct observables\n\n")
        fh.write(f"Sample N = {N} (same quality cuts as canonical).\n\n")
        fh.write("| observable | n | spearman rho | p_perm | 95% CI |\n")
        fh.write("|---|---:|---:|---:|---|\n")
        for row in primary_rows:
            fh.write(f"| {row['obs']} | {row['n']} | "
                     f"{row['spearman_r']:+.3f} | {row['perm_p']:.4f} | "
                     f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] |\n")
        fh.write("\n")

        fh.write("## Population-level test: alpha > 0\n\n")
        fh.write(f"- Pointer log-enhancement alpha, n={len(alpha)}.\n")
        fh.write(f"- median={np.median(alpha):.3f}, frac>0.1={np.mean(alpha>0.1)*100:.0f}%, "
                 f"frac>=4.9={np.mean(alpha>=4.9)*100:.0f}%.\n")
        fh.write(f"- One-sided Wilcoxon (H1: median alpha > 0): W={w_stat}, p={w_p:.2e}.\n")
    print("\n  wrote phase2_report.md")


if __name__ == "__main__":
    main()
