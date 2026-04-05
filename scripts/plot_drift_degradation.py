#!/usr/bin/env python3
"""
Build degradation-focused figures from eval_drift sweep CSVs.

Reads drift_eval_grid.csv (with degradation = 1 - IQM/baseline).
  - Positive degradation  => IQM below baseline (hurt vs stationary σ=0).
  - Negative degradation  => IQM above baseline (better than baseline under drift).

Outputs a multi-panel PNG for stress_sigma and stress_theta sweeps.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _pivot(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Z (sigma rows, theta cols), sigma list, theta list."""
    sigmas = np.sort(df["sigma"].unique())
    thetas = np.sort(df["theta"].unique())
    z = np.full((len(sigmas), len(thetas)), np.nan)
    for _, row in df.iterrows():
        i = int(np.where(sigmas == row["sigma"])[0][0])
        j = int(np.where(thetas == row["theta"])[0][0])
        z[i, j] = row["degradation"]
    return z, sigmas, thetas


def _bin_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges for pcolormesh from sorted bin centers."""
    c = np.asarray(centers, dtype=float)
    e = np.empty(len(c) + 1)
    e[0] = c[0] - (c[1] - c[0]) / 2
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    e[1:-1] = (c[:-1] + c[1:]) / 2
    return e


def _panel_heatmap(ax, z, sigmas, thetas, title: str, threshold: float = 0.10):
    vmax = float(np.nanmax(np.abs(z)))
    vmax = max(vmax, threshold * 1.5, 0.15)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    Te, Se = _bin_edges(thetas), _bin_edges(sigmas)
    X, Y = np.meshgrid(Te, Se)
    im = ax.pcolormesh(X, Y, z, cmap="RdBu_r", norm=norm, shading="flat")
    ax.set_xlabel(r"$\theta$ (mean reversion)")
    ax.set_ylabel(r"$\sigma$ (OU noise)")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Degradation (1 − IQM / baseline)\n+ = worse than σ=0 baseline")
    try:
        TT, SS = np.meshgrid(thetas, sigmas)
        cs = ax.tricontour(
            TT.ravel(),
            SS.ravel(),
            z.ravel(),
            levels=[threshold],
            colors="black",
            linewidths=1.5,
            linestyles="--",
        )
        ax.clabel(cs, inline=True, fontsize=8)
    except Exception:
        pass


def _panel_lines_sigma(ax, df: pd.DataFrame, title: str, threshold: float):
    thetas = np.sort(df["theta"].unique())
    for th in thetas:
        sub = df[df["theta"] == th].sort_values("sigma")
        ax.plot(sub["sigma"], sub["degradation"], marker="o", ms=3, label=f"θ={th:g}")
    ax.axhline(threshold, color="black", ls="--", lw=1, label=f"{threshold:.0%} hurt line")
    ax.axhline(0, color="gray", ls="-", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Degradation")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)


def _panel_lines_theta(ax, df: pd.DataFrame, title: str, threshold: float):
    sigmas = np.sort(df["sigma"].unique())
    for sg in sigmas:
        sub = df[df["sigma"] == sg].sort_values("theta")
        ax.plot(sub["theta"], sub["degradation"], marker="o", ms=3, label=f"σ={sg:g}")
    ax.axhline(threshold, color="black", ls="--", lw=1, label=f"{threshold:.0%} hurt line")
    ax.axhline(0, color="gray", ls="-", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Degradation")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)


def plot_pair(csv_path: str, label: str, out_path: str, threshold: float):
    df = pd.read_csv(csv_path)
    if "degradation" not in df.columns:
        raise SystemExit(f"No degradation column in {csv_path}")

    z, sigmas, thetas = _pivot(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    _panel_heatmap(
        axes[0],
        z,
        sigmas,
        thetas,
        title=f"{label}: degradation heatmap\n(blue = better than baseline, red = worse)",
        threshold=threshold,
    )

    if "stress_sigma" in label.lower() or "sigma" in os.path.basename(os.path.dirname(csv_path)):
        _panel_lines_sigma(
            axes[1],
            df,
            title=f"{label}: slices at fixed θ",
            threshold=threshold,
        )
    else:
        _panel_lines_theta(
            axes[1],
            df,
            title=f"{label}: slices at fixed σ",
            threshold=threshold,
        )

    fig.suptitle(
        f"{label}  |  baseline = stationary σ=0 (IQM averaged over θ at σ=0)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_combined(sigma_csv: str, theta_csv: str, out_path: str, threshold: float):
    """2x2: both heatmaps + both line-style summaries."""
    dfs = {
        "stress_sigma": pd.read_csv(sigma_csv),
        "stress_theta": pd.read_csv(theta_csv),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for row, (name, df) in enumerate(dfs.items()):
        z, sigmas, thetas = _pivot(df)
        _panel_heatmap(
            axes[row, 0],
            z,
            sigmas,
            thetas,
            title=f"{name}: heatmap",
            threshold=threshold,
        )
        if name == "stress_sigma":
            _panel_lines_sigma(axes[row, 1], df, title=f"{name}: vs σ", threshold=threshold)
        else:
            _panel_lines_theta(axes[row, 1], df, title=f"{name}: vs θ", threshold=threshold)

    fig.suptitle(
        "GPL drift eval — degradation (1 − IQM / baseline)  |  + = worse than σ=0 baseline",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot drift degradation from sweep CSVs")
    p.add_argument(
        "--sigma-csv",
        default="results/eval_drift_sweep_stress_sigma/drift_eval_grid.csv",
    )
    p.add_argument(
        "--theta-csv",
        default="results/eval_drift_sweep_stress_theta/drift_eval_grid.csv",
    )
    p.add_argument(
        "--out-dir",
        default="results/eval_drift_figures",
        help="Directory for output PNGs",
    )
    p.add_argument("--threshold", type=float, default=0.10, help="Hurt threshold (fraction)")
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(root)

    out_dir = args.out_dir
    thr = args.threshold

    if os.path.isfile(args.sigma_csv):
        plot_pair(
            args.sigma_csv,
            "stress_sigma",
            os.path.join(out_dir, "degradation_stress_sigma.png"),
            thr,
        )
    else:
        print(f"[skip] missing {args.sigma_csv}", file=sys.stderr)

    if os.path.isfile(args.theta_csv):
        plot_pair(
            args.theta_csv,
            "stress_theta",
            os.path.join(out_dir, "degradation_stress_theta.png"),
            thr,
        )
    else:
        print(f"[skip] missing {args.theta_csv}", file=sys.stderr)

    if os.path.isfile(args.sigma_csv) and os.path.isfile(args.theta_csv):
        plot_combined(
            args.sigma_csv,
            args.theta_csv,
            os.path.join(out_dir, "degradation_combined_2x2.png"),
            thr,
        )


if __name__ == "__main__":
    main()
