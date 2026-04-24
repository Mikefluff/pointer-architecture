#!/usr/bin/env python3
"""
Phase 4 — Residual structure after Burkert subtraction.

Rationale:
  If the archive interpretation of dark matter is correct, residuals left
  after fitting the simplest well-behaved halo (Burkert, which already wins
  SPARC AIC in the canonical run) should carry a systematic,
  evolution-linked signal. They should also be more structured than one
  would expect from white measurement noise.

  This module:
    1. Refits Burkert per galaxy (fast — 4 params, ~1 s each).
    2. Computes several structural features of the Burkert residuals
       r = (V_obs − V_model_burkert) / errV.
    3. Correlates each feature with the composite maturity score from
       Phase 2, applying Benjamini-Hochberg FDR correction across features.

Features:
  - resid_rms                     : √⟨r²⟩
  - resid_mean_abs                : ⟨|r|⟩
  - resid_radial_slope            : linear slope r vs. radius (raw)
  - resid_radial_slope_spearman   : rank slope
  - resid_compactness             : |⟨r_inner_half⟩| − |⟨r_outer_half⟩|
  - resid_spectral_slope          : log-power vs log-freq slope of FFT(r)

Inputs:
  - phase2_derived.csv   (for composite_age + quality gates)
  - SPARC rotmod and Table 1

Outputs:
  - phase4_residuals.csv
  - phase4_report.md
"""

from __future__ import annotations

import csv
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import rankdata

import analysis as base


def fit_burkert_single(r, Vobs, errV, Vgas, Vdisk, Vbul,
                       has_bulge: bool, Rdisk: float) -> Tuple[np.ndarray, float]:
    """Refit Burkert halo only. Re-uses methodology_final.fit_model."""
    out = base.fit_model("burkert", r, Vobs, errV, Vgas, Vdisk, Vbul,
                         has_bulge=has_bulge, Rdisk=max(Rdisk, 0.1))
    return out.params, out.chi2


def burkert_model_curve(r, y_disk, y_bul, rho_c, r_c,
                        Vgas, Vdisk, Vbul) -> np.ndarray:
    v2halo = base.v2_burkert(r, rho_c, r_c)
    return base.v_total(r, Vgas, Vdisk, Vbul, y_disk, y_bul, v2halo)


def extract_features(r: np.ndarray, z: np.ndarray) -> dict:
    """Structural features of residual vector z = (V_obs − V_model) / errV."""
    if len(z) < 5:
        return {k: float("nan") for k in (
            "rms", "mean_abs", "radial_slope", "radial_slope_spearman",
            "compactness", "spectral_slope")}

    rms = float(np.sqrt(np.mean(z ** 2)))
    mean_abs = float(np.mean(np.abs(z)))

    # linear slope of z against r (raw)
    dr = r - r.mean(); dz = z - z.mean()
    denom = float((dr * dr).sum())
    slope = float((dr * dz).sum() / denom) if denom > 0 else float("nan")

    # Spearman slope: correlation rank_r vs rank_z
    rr = rankdata(r); rz = rankdata(z)
    n = len(r)
    dr2 = rr - rr.mean(); dz2 = rz - rz.mean()
    denom2 = math.sqrt((dr2 * dr2).sum() * (dz2 * dz2).sum())
    rho = float((dr2 * dz2).sum() / denom2) if denom2 > 0 else float("nan")

    # compactness: mean(|z| inner half) - mean(|z| outer half)
    half = len(r) // 2
    inner = z[:half]; outer = z[half:]
    compactness = float(np.mean(np.abs(inner)) - np.mean(np.abs(outer))) \
        if len(inner) and len(outer) else float("nan")

    # spectral slope from |FFT(z)|^2 vs frequency (log-log)
    if len(z) >= 8:
        fft = np.fft.rfft(z - z.mean())
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(z), d=1.0)
        # skip DC (freq 0)
        mask = (freqs > 0) & (power > 0)
        if mask.sum() >= 3:
            lx = np.log10(freqs[mask]); ly = np.log10(power[mask])
            dx = lx - lx.mean(); dy = ly - ly.mean()
            d2 = float((dx * dx).sum())
            spec = float((dx * dy).sum() / d2) if d2 > 0 else float("nan")
        else:
            spec = float("nan")
    else:
        spec = float("nan")

    return dict(rms=rms, mean_abs=mean_abs,
                radial_slope=slope, radial_slope_spearman=rho,
                compactness=compactness, spectral_slope=spec)


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    rx = rankdata(x); ry = rankdata(y)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    dx = rx - rx.mean(); dy = ry - ry.mean()
    denom = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if denom <= 0:
        return float("nan"), float("nan")
    r = float((dx * dy).sum() / denom)
    r = max(-0.9999, min(0.9999, r))
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def permutation_p(x, y, n_perm=10000, seed=0):
    rng = np.random.RandomState(seed)
    r_obs = spearman(x, y)[0]
    if not np.isfinite(r_obs):
        return float("nan")
    y = np.asarray(y, dtype=float)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r = spearman(x, yp)[0]
        if np.isfinite(r) and abs(r) >= abs(r_obs):
            count += 1
    return count / n_perm


def bh_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR-corrected q-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    q_raw = ranked * n / (np.arange(n) + 1)
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    out = np.empty(n)
    out[order] = q_mono
    return [float(x) for x in out]


def main():
    print("Phase 4 — residual structure after Burkert subtraction")

    # load composite_age from phase 2
    composite: Dict[str, float] = {}
    phase2_rows: Dict[str, dict] = {}
    with open("phase2_derived.csv") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            composite[row["name"]] = float(row["composite_age"])
            phase2_rows[row["name"]] = row
    print(f"  loaded composite_age for {len(composite)} galaxies")

    galaxies = base.parse_table1()
    curves = base.load_rotcurves()

    rows_out: List[dict] = []
    for i, name in enumerate(composite):
        g = galaxies[name]
        d = curves[name.replace(" ", "")]
        has_bulge = bool(np.any(np.abs(d["Vbul"]) > 1.0))

        params, chi2 = fit_burkert_single(d["Rad"], d["Vobs"], d["errV"],
                                          d["Vgas"], d["Vdisk"], d["Vbul"],
                                          has_bulge=has_bulge, Rdisk=g["Rdisk"])
        Y_disk, Y_bul, rho_c, r_c = [float(x) for x in params]
        V_model = burkert_model_curve(d["Rad"], Y_disk, Y_bul, rho_c, r_c,
                                      d["Vgas"], d["Vdisk"], d["Vbul"])
        z = (d["Vobs"] - V_model) / d["errV"]
        feats = extract_features(d["Rad"], z)
        rows_out.append(dict(
            name=name,
            chi2_burkert=chi2,
            Y_disk=Y_disk, Y_bul=Y_bul, rho_c=rho_c, r_c=r_c,
            composite_age=composite[name],
            **{f"resid_{k}": v for k, v in feats.items()},
        ))
        if (i + 1) % 25 == 0 or i == len(composite) - 1:
            print(f"  {i+1:3d}/{len(composite)}  {name:<14}  "
                  f"chi2={chi2:5.2f}  rms={feats['rms']:.2f}  "
                  f"slope={feats['radial_slope']:+.3f}")

    # write csv
    keys = ["name", "chi2_burkert", "Y_disk", "Y_bul", "rho_c", "r_c",
            "composite_age",
            "resid_rms", "resid_mean_abs", "resid_radial_slope",
            "resid_radial_slope_spearman", "resid_compactness",
            "resid_spectral_slope"]
    with open("phase4_residuals.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows_out:
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
    print(f"  wrote phase4_residuals.csv ({len(rows_out)} rows)")

    # correlation feature vs composite_age, FDR-corrected
    # quality: chi2_burkert<10
    good = [r for r in rows_out if np.isfinite(r["composite_age"])
            and np.isfinite(r["resid_rms"]) and r["chi2_burkert"] < 10]
    N = len(good)
    print(f"\n[correlations] N = {N}")
    age = np.array([r["composite_age"] for r in good])
    feat_keys = ["resid_rms", "resid_mean_abs", "resid_radial_slope",
                 "resid_radial_slope_spearman", "resid_compactness",
                 "resid_spectral_slope"]
    corr_rows = []
    p_list = []
    for key in feat_keys:
        y = np.array([r[key] for r in good])
        mask = np.isfinite(y)
        if mask.sum() < 20:
            corr_rows.append(dict(feat=key, n=int(mask.sum()),
                                  rho=float("nan"), p=float("nan"),
                                  p_perm=float("nan")))
            p_list.append(1.0)
            continue
        rho, p_asy = spearman(age[mask], y[mask])
        p_perm = permutation_p(age[mask], y[mask], n_perm=10000, seed=101)
        corr_rows.append(dict(feat=key, n=int(mask.sum()),
                              rho=rho, p=p_asy, p_perm=p_perm))
        p_list.append(p_perm)
    q_vals = bh_fdr(p_list)
    for i, row in enumerate(corr_rows):
        row["q_fdr"] = q_vals[i]
        print(f"  {row['feat']:<28}  n={row['n']}  rho={row['rho']:+.3f}  "
              f"p_perm={row['p_perm']:.4f}  q_fdr={row['q_fdr']:.4f}")

    # report
    with open("phase4_report.md", "w") as fh:
        fh.write("# Phase 4 — Residual structure after Burkert subtraction\n\n")
        fh.write(f"Sample: {len(rows_out)} galaxies refit with Burkert halo; "
                 f"{N} pass chi2<10 cut for the correlation test.\n\n")
        fh.write("## Feature correlations with composite age (Spearman)\n\n")
        fh.write("| feature | n | rho | p_perm | q (FDR, BH) |\n")
        fh.write("|---|---:|---:|---:|---:|\n")
        for row in corr_rows:
            fh.write(f"| {row['feat']} | {row['n']} | "
                     f"{row['rho']:+.3f} | {row['p_perm']:.4f} | "
                     f"{row['q_fdr']:.4f} |\n")
        fh.write("\nPositive composite_age = more evolved galaxy. "
                 "Features with q_fdr < 0.05 survive multiplicity.\n")
    print("  wrote phase4_report.md")


if __name__ == "__main__":
    main()
