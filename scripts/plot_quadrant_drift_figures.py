#!/usr/bin/env python3
"""
Build degradation visuals for paper quadrants (Q1 legacy vs Q4 nerfed-policy).

Reuses plot_drift_degradation.plot_pair / plot_combined.
Runs analyze_capability_confound for episode-level regressions (optional).

Q1 — pre–paper-LBF-policy drift evals (legacy result trees; old checkpoint at eval time).
Q4 — paper-LBF–trained policy drift bundle (eval_drift_policy_nerfed128k/).

Q2 / Q3 — not materialized as separate result dirs in this repo (skip plots).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

import importlib.util

# Repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_pd_path = os.path.join(ROOT, "scripts", "plot_drift_degradation.py")
_spec = importlib.util.spec_from_file_location("_plot_drift_degradation", _pd_path)
_pd_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pd_mod)
plot_pair = _pd_mod.plot_pair
plot_combined = _pd_mod.plot_combined


# Paths relative to repo root
Q1 = {
    "label": "Q1_legacy_policy",
    "main": "results/eval_drift_sweep_main/drift_eval_grid.csv",
    "main_ext": "results/eval_drift_sweep_main_extended/drift_eval_grid.csv",
    "coupled": "results/eval_drift_sweep_coupled/drift_eval_grid.csv",
    "stress_sigma": "results/eval_drift_sweep_stress_sigma/drift_eval_grid.csv",
    "stress_theta": "results/eval_drift_sweep_stress_theta/drift_eval_grid.csv",
    "episodes_main": "results/eval_drift_sweep_main/drift_eval_episodes.csv",
}

Q4 = {
    "label": "Q4_policy_nerfed128k",
    "main": "results/eval_drift_policy_nerfed128k/sweep_main/drift_eval_grid.csv",
    "main_ext": "results/eval_drift_policy_nerfed128k/sweep_main_extended/drift_eval_grid.csv",
    "coupled": "results/eval_drift_policy_nerfed128k/sweep_coupled/drift_eval_grid.csv",
    "stress_sigma": "results/eval_drift_policy_nerfed128k/bounds_stress_sigma/drift_eval_grid.csv",
    "stress_theta": "results/eval_drift_policy_nerfed128k/bounds_stress_theta/drift_eval_grid.csv",
    "episodes_main": "results/eval_drift_policy_nerfed128k/sweep_main/drift_eval_episodes.csv",
}


def _ensure_degradation_csv(csv_path: str, cache_dir: str) -> str:
    """Return path to CSV with degradation column (add if missing, e.g. baseline IQM=0)."""
    df = pd.read_csv(csv_path)
    if "degradation" in df.columns:
        return csv_path
    if "iqm_return" not in df.columns:
        return csv_path
    base = float(df.loc[df["sigma"] == 0.0, "iqm_return"].mean())
    if base > 1e-9:
        df["degradation"] = 1.0 - df["iqm_return"] / base
    else:
        base_m = float(df.loc[df["sigma"] == 0.0, "mean_return"].mean())
        if base_m > 1e-9:
            df["degradation"] = 1.0 - df["mean_return"] / base_m
        else:
            df["degradation"] = 0.0
    os.makedirs(cache_dir, exist_ok=True)
    base_name = os.path.basename(csv_path).replace(".csv", "")
    out_csv = os.path.join(cache_dir, f"{base_name}_with_degradation.csv")
    df.to_csv(out_csv, index=False)
    return out_csv


def _safe_plot_pair(csv_path: str, tag: str, out_path: str, thr: float, cache_dir: str) -> None:
    if not os.path.isfile(csv_path):
        print(f"[skip] missing {csv_path}", file=sys.stderr)
        return
    path_use = _ensure_degradation_csv(csv_path, cache_dir)
    plot_pair(path_use, tag, out_path, thr)
    print(f"OK {out_path}")


def _safe_combined(sig_csv: str, th_csv: str, out_path: str, thr: float, cache_dir: str) -> None:
    if not os.path.isfile(sig_csv) or not os.path.isfile(th_csv):
        print(f"[skip combined] missing {sig_csv} or {th_csv}", file=sys.stderr)
        return
    sig_u = _ensure_degradation_csv(sig_csv, cache_dir)
    th_u = _ensure_degradation_csv(th_csv, cache_dir)
    plot_combined(sig_u, th_u, out_path, thr)
    print(f"OK {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        default="results/eval_drift_figures_quadrants",
        help="Output directory for PNGs",
    )
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument(
        "--skip-capability",
        action="store_true",
        help="Do not run analyze_capability_confound.py",
    )
    args = p.parse_args()
    os.chdir(ROOT)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    thr = args.threshold
    cache_dir = os.path.join(out, "_grid_cache")

    # --- Q1 ---
    _safe_plot_pair(
        os.path.join(ROOT, Q1["main"]),
        "Q1_main_canonical",
        os.path.join(out, "q1_main_canonical_degradation.png"),
        thr,
        cache_dir,
    )
    _safe_plot_pair(
        os.path.join(ROOT, Q1["coupled"]),
        "Q1_coupled_canonical",
        os.path.join(out, "q1_coupled_canonical_degradation.png"),
        thr,
        cache_dir,
    )
    _safe_combined(
        os.path.join(ROOT, Q1["stress_sigma"]),
        os.path.join(ROOT, Q1["stress_theta"]),
        os.path.join(out, "q1_stress_sigma_theta_combined_2x2.png"),
        thr,
        cache_dir,
    )
    _safe_plot_pair(
        os.path.join(ROOT, Q1["main_ext"]),
        "Q1_main_extended_sigma",
        os.path.join(out, "q1_main_extended_degradation.png"),
        thr,
        cache_dir,
    )

    # --- Q4 ---
    _safe_plot_pair(
        os.path.join(ROOT, Q4["main"]),
        "Q4_main_canonical",
        os.path.join(out, "q4_main_canonical_degradation.png"),
        thr,
        cache_dir,
    )
    _safe_plot_pair(
        os.path.join(ROOT, Q4["coupled"]),
        "Q4_coupled_canonical",
        os.path.join(out, "q4_coupled_canonical_degradation.png"),
        thr,
        cache_dir,
    )
    _safe_combined(
        os.path.join(ROOT, Q4["stress_sigma"]),
        os.path.join(ROOT, Q4["stress_theta"]),
        os.path.join(out, "q4_stress_sigma_theta_combined_2x2.png"),
        thr,
        cache_dir,
    )
    _safe_plot_pair(
        os.path.join(ROOT, Q4["main_ext"]),
        "Q4_main_extended_sigma",
        os.path.join(out, "q4_main_extended_degradation.png"),
        thr,
        cache_dir,
    )

    if not args.skip_capability:
        pairs = [
            (
                os.path.join(ROOT, Q1["episodes_main"]),
                os.path.join(out, "q1_capability_confound"),
            ),
            (
                os.path.join(ROOT, Q4["episodes_main"]),
                os.path.join(out, "q4_capability_confound"),
            ),
        ]
        for ep_csv, cap_out in pairs:
            if not os.path.isfile(ep_csv):
                print(f"[skip capability] missing {ep_csv}", file=sys.stderr)
                continue
            os.makedirs(cap_out, exist_ok=True)
            cmd = [
                sys.executable,
                os.path.join(ROOT, "experiments", "analyze_capability_confound.py"),
                "--episodes-csv",
                ep_csv,
                "--out-dir",
                cap_out,
            ]
            print("RUN", " ".join(cmd))
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
