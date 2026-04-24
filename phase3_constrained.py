#!/usr/bin/env python3
"""
Phase 3 — Constrained Pointer Architecture (α and r_core as population relations).

Rationale:
  The canonical six-parameter Pointer model loses AIC against Burkert
  (13/171 vs 103/171). But the theory does not actually predict that α
  and r_core are galaxy-by-galaxy free parameters: α is the strength
  of the log-enhancement, which the accumulation hypothesis ties to how
  much history a galaxy has had (∝ N_orbits); r_core is a small inner
  scale that should track the disk length.

  We therefore fit two population relations across all 171 galaxies
  using the canonical free-parameter pointer fits from Phase 2:
      α       = a + b · log10(N_orbits)
      r_core  = c · R_disk                   (one coefficient)
  Then we refit each galaxy with α and r_core *fixed* by these relations.
  The constrained pointer model has 4 per-galaxy parameters
  (Y_disk, Y_bul, ρ0, r_mem), matching NFW and Burkert for a fair
  AIC/BIC comparison.

Inputs:
  - phase2_derived.csv   (free-fit α, r_core for each galaxy)
  - SPARC_Lelli2016c.mrt
  - Rotmod_LTG/

Outputs:
  - phase3_constrained.csv     per-galaxy chi2 + AIC + BIC + constrained ρ0, r_mem
  - phase3_report.md
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize

import analysis as base

G_ASTRO = base.G_ASTRO


# ------------------------------------------------------------
# Constrained fit
# ------------------------------------------------------------

def v2_pointer_fixed(r, rho0, r_mem, alpha, r_core):
    return base.v2_pointer(r, rho0, r_mem, alpha, r_core)


def fit_pointer_constrained(r, Vobs, errV, Vgas, Vdisk, Vbul,
                            has_bulge: bool, Rdisk: float,
                            alpha_fixed: float, r_core_fixed: float,
                            seed: int = 42) -> Tuple[np.ndarray, float]:
    """4-parameter fit: Y_disk, Y_bul, log10(rho0), log10(r_mem). alpha, r_core fixed."""
    yb = (0.0, 2.0) if has_bulge else (0.0, 0.01)
    Rmax = float(np.max(r))
    log_r_hi = math.log10(max(200.0, 4.0 * Rmax))
    bounds = [(0.1, 1.5), yb, (3.0, 11.0), (-1.0, log_r_hi)]

    def chi2_fun(p):
        Y_disk, Y_bul, log_rho0, log_r_mem = p
        rho0 = 10 ** log_rho0
        r_mem = 10 ** log_r_mem
        v2halo = v2_pointer_fixed(r, rho0, r_mem, alpha_fixed, r_core_fixed)
        vmodel = base.v_total(r, Vgas, Vdisk, Vbul, Y_disk, Y_bul, v2halo)
        return float(np.sum(((Vobs - vmodel) / errV) ** 2) / max(len(r) - len(p), 1))

    de = differential_evolution(chi2_fun, bounds, seed=seed,
                                maxiter=600, tol=1e-7, polish=True,
                                popsize=20, init="sobol")
    best_p, best_chi2 = de.x, de.fun

    for y_d in (0.3, 0.5, 0.7):
        for scale_mult in (0.5, 2.0, 8.0):
            x0 = [y_d, 0.5 if has_bulge else 0.0,
                  7.0, math.log10(max(0.1, Rdisk * scale_mult))]
            try:
                res = minimize(chi2_fun, x0, method="Nelder-Mead",
                               options={"maxiter": 3000})
                in_b = all(bounds[i][0] <= res.x[i] <= bounds[i][1] for i in range(4))
                if in_b and res.fun < best_chi2:
                    best_chi2, best_p = res.fun, res.x
            except Exception:
                continue

    physical = np.array([best_p[0], best_p[1], 10 ** best_p[2], 10 ** best_p[3]])
    return physical, best_chi2


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("Phase 3 — constrained Pointer fit")

    galaxies = base.parse_table1()
    curves = base.load_rotcurves()

    # Read free-fit pointer params from phase 2
    free_rows: Dict[str, dict] = {}
    with open("phase2_derived.csv") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            free_rows[row["name"]] = dict(
                alpha=float(row["pointer_alpha"]),
                r_core=float(row["pointer_r_core"]),
                r_mem=float(row["pointer_r_mem"]),
                rho0=float(row["pointer_rho0"]),
                T=int(row["T"]),
                Rdisk=float(row["Rdisk"]),
                Vflat=float(row["Vflat"]),
                Qual=int(row["Qual"]),
                chi2_free=float(row["pointer_chi2"]),
            )
    print(f"  loaded {len(free_rows)} free-fit pointer rows")

    # Compute log_N_orbits for each galaxy
    log_Norb: Dict[str, float] = {}
    for name in free_rows:
        g = galaxies[name]
        if g["Vflat"] > 0 and g["Rdisk"] > 0:
            t_orb_Gyr = 2.0 * math.pi * g["Rdisk"] / g["Vflat"] * 0.978
            N = 13.7 / max(t_orb_Gyr, 1e-3)
            log_Norb[name] = math.log10(N) if N > 0 else float("nan")
        else:
            log_Norb[name] = float("nan")

    # Fit population relation: alpha ~ a + b*log_N_orbits
    # Filter galaxies whose free alpha is not pinned at bounds.
    names_for_relation = []
    alphas, log_norbs, r_cores, Rdisks = [], [], [], []
    for name, row in free_rows.items():
        if not np.isfinite(log_Norb.get(name, float("nan"))):
            continue
        if row["alpha"] >= 4.9 or row["alpha"] <= 0.001:
            continue  # skip galaxies pinned at bounds, they don't constrain
        names_for_relation.append(name)
        alphas.append(row["alpha"])
        log_norbs.append(log_Norb[name])
        r_cores.append(row["r_core"])
        Rdisks.append(row["Rdisk"])
    alphas = np.array(alphas); log_norbs = np.array(log_norbs)
    r_cores = np.array(r_cores); Rdisks = np.array(Rdisks)
    print(f"  {len(alphas)} galaxies have free-fit alpha not at bounds")

    # Linear regression alpha ~ a + b log_N
    A = np.vstack([np.ones_like(log_norbs), log_norbs]).T
    coef_alpha, *_ = np.linalg.lstsq(A, alphas, rcond=None)
    a_coef, b_coef = float(coef_alpha[0]), float(coef_alpha[1])
    alpha_pred = A @ coef_alpha
    alpha_rmse = float(np.sqrt(np.mean((alphas - alpha_pred) ** 2)))
    alpha_r = float(np.corrcoef(log_norbs, alphas)[0, 1])
    print(f"  alpha ≈ {a_coef:+.3f} + {b_coef:+.3f} · log10(N_orbits)  "
          f"(r={alpha_r:+.3f}, rmse={alpha_rmse:.3f})")

    # r_core ≈ c · Rdisk (force through origin, since physically r_core=0 when Rdisk=0)
    c_coef = float(np.sum(r_cores * Rdisks) / max(np.sum(Rdisks ** 2), 1e-10))
    rcore_pred = c_coef * Rdisks
    rcore_rmse = float(np.sqrt(np.mean((r_cores - rcore_pred) ** 2)))
    rcore_r = float(np.corrcoef(Rdisks, r_cores)[0, 1])
    print(f"  r_core ≈ {c_coef:+.3f} · R_disk  "
          f"(r={rcore_r:+.3f}, rmse={rcore_rmse:.3f})")

    # Refit each galaxy with alpha, r_core constrained
    print(f"\n  refitting {len(free_rows)} galaxies with 4 free params each")
    results: List[dict] = []
    for i, name in enumerate(free_rows):
        row = free_rows[name]
        g = galaxies[name]; d = curves[name.replace(" ", "")]
        has_bulge = bool(np.any(np.abs(d["Vbul"]) > 1.0))

        # constraints
        lnN = log_Norb.get(name, float("nan"))
        if np.isfinite(lnN):
            alpha_c = float(np.clip(a_coef + b_coef * lnN, 0.0, 5.0))
        else:
            alpha_c = float(np.clip(a_coef, 0.0, 5.0))
        r_core_c = float(np.clip(c_coef * g["Rdisk"], 0.01, 50.0))

        phys, chi2 = fit_pointer_constrained(
            d["Rad"], d["Vobs"], d["errV"], d["Vgas"], d["Vdisk"], d["Vbul"],
            has_bulge=has_bulge, Rdisk=max(g["Rdisk"], 0.1),
            alpha_fixed=alpha_c, r_core_fixed=r_core_c)
        n = len(d["Rad"]); k = 4
        chi2_abs = chi2 * max(n - k, 1)
        aic = chi2_abs + 2 * k
        bic = chi2_abs + k * math.log(max(n, 2))
        results.append(dict(
            name=name, n=n, k=k,
            alpha_fixed=alpha_c, r_core_fixed=r_core_c,
            Y_disk=float(phys[0]), Y_bul=float(phys[1]),
            rho0=float(phys[2]), r_mem=float(phys[3]),
            chi2_constrained=chi2, aic=aic, bic=bic,
            chi2_free=row["chi2_free"],
        ))
        if (i + 1) % 20 == 0 or i == len(free_rows) - 1:
            print(f"  {i+1:3d}/{len(free_rows)}  {name:<14}  "
                  f"chi2 free={row['chi2_free']:5.2f}  constrained={chi2:5.2f}  "
                  f"α={alpha_c:.2f}  r_core={r_core_c:.2f}")

    # write
    with open("phase3_constrained.csv", "w") as fh:
        keys = ["name", "n", "k",
                "alpha_fixed", "r_core_fixed",
                "Y_disk", "Y_bul", "rho0", "r_mem",
                "chi2_constrained", "aic", "bic", "chi2_free"]
        fh.write(",".join(keys) + "\n")
        for r in results:
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
    print(f"  wrote phase3_constrained.csv")

    # summary vs Burkert/NFW from final_fits.csv
    other_aic: Dict[str, Tuple[float, float]] = {}
    with open("final_fits.csv") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            other_aic[row["name"]] = (float(row["nfw_aic"]), float(row["burkert_aic"]))

    chi2s = np.array([r["chi2_constrained"] for r in results])
    aics = np.array([r["aic"] for r in results])

    wins = {"pointer_c": 0, "nfw": 0, "burkert": 0}
    for r in results:
        nfw_a, burk_a = other_aic[r["name"]]
        amap = {"pointer_c": r["aic"], "nfw": nfw_a, "burkert": burk_a}
        wins[min(amap, key=amap.get)] += 1

    print("\n[summary] constrained pointer vs NFW / Burkert (all k=4)")
    print(f"  median chi2 constrained = {np.median(chi2s):.3f}")
    print(f"  median AIC constrained = {np.median(aics):.3f}")
    print(f"  AIC best-model split: pointer_c={wins['pointer_c']}, "
          f"nfw={wins['nfw']}, burkert={wins['burkert']}")

    with open("phase3_report.md", "w") as fh:
        fh.write("# Phase 3 — Constrained Pointer Architecture\n\n")
        fh.write("## Population relations\n\n")
        fh.write(f"Using {len(alphas)} galaxies with free alpha not pinned at bounds:\n\n")
        fh.write(f"- alpha  ≈ {a_coef:+.3f} + {b_coef:+.3f} · log10(N_orbits)   "
                 f"(r={alpha_r:+.3f}, RMSE={alpha_rmse:.3f})\n")
        fh.write(f"- r_core ≈ {c_coef:+.3f} · R_disk   "
                 f"(r={rcore_r:+.3f}, RMSE={rcore_rmse:.3f})\n\n")
        fh.write("## Refit with alpha, r_core fixed (k=4 per galaxy)\n\n")
        fh.write(f"- median chi2 (constrained pointer) = {np.median(chi2s):.3f}\n")
        fh.write(f"- median AIC  (constrained pointer) = {np.median(aics):.3f}\n\n")
        fh.write(f"AIC best model, head-to-head at k=4:\n\n")
        fh.write(f"- Pointer (constrained): {wins['pointer_c']}\n")
        fh.write(f"- NFW: {wins['nfw']}\n")
        fh.write(f"- Burkert: {wins['burkert']}\n\n")
        fh.write("Compare to the free-parameter canonical run: pointer wins AIC "
                 "on 13/171 with k=6. The constrained variant matches Burkert "
                 "and NFW in parameter count.\n")
    print("  wrote phase3_report.md")


if __name__ == "__main__":
    main()
