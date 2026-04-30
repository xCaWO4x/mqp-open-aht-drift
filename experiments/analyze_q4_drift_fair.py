#!/usr/bin/env python3
"""
Summarize each drift sweep on its own absolute performance (no cross-policy pairing).

Reads drift_eval_grid.csv per run and reports distribution of iqm_return and mean_return
across the (sigma, theta) grid. Optional bootstrap CI on mean iqm_return over cells.

Usage:
  python experiments/analyze_q4_drift_fair.py
  python experiments/analyze_q4_drift_fair.py --results-root results
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from dataclasses import dataclass
from statistics import median, stdev
from typing import Dict, List, Optional, Tuple


@dataclass
class Grid:
    name: str
    path: str
    rows: List[dict]


def load_grid(path: str, name: str) -> Grid:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    key = lambda r: (float(r["sigma"]), float(r["theta"]))
    rows.sort(key=key)
    return Grid(name=name, path=path, rows=rows)


def scalar_stats(vals: List[float]) -> Dict[str, float]:
    vals_s = sorted(vals)
    n = len(vals_s)

    def pct(p: float) -> float:
        if n == 0:
            return float("nan")
        k = int(round((p / 100.0) * (n - 1)))
        return vals_s[k]

    return {
        "n": float(n),
        "mean": sum(vals) / n if n else float("nan"),
        "median": median(vals) if n else float("nan"),
        "stdev": stdev(vals) if n > 1 else 0.0,
        "min": min(vals) if n else float("nan"),
        "max": max(vals) if n else float("nan"),
        "p90": pct(90),
        "p95": pct(95),
    }


def bootstrap_mean_ci(vals: List[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return lo, hi


def read_baseline_iqm(results_root: str, folder: str) -> Optional[float]:
    p = os.path.join(results_root, folder, "baseline_summary.txt")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        for line in f:
            m = re.match(r"baseline_iqm:\s*([0-9.eE+-]+)", line.strip())
            if m:
                return float(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Bootstrap resamples for 95%% CI on mean iqm (0=skip).",
    )
    args = ap.parse_args()

    root = args.results_root
    default_runs = [
        ("q2_baseline_drift", "Q2"),
        ("q4_hardened_drift", "Q4_hardened"),
        ("q4_rw_drift", "Q4_rw"),
        ("q4_inf_rw_drift", "Q4_inf"),
        ("q4_inf_aux_rw_drift", "Q4_inf_aux"),
        ("q4_inf_ema_rw_drift", "Q4_inf_ema"),
    ]
    folder_by_label = {label: folder for folder, label in default_runs}

    grids: Dict[str, Grid] = {}
    for folder, label in default_runs:
        p = os.path.join(root, folder, "drift_eval_grid.csv")
        if os.path.isfile(p):
            grids[label] = load_grid(p, label)

    lines: List[str] = []
    lines.append("Drift sweep — per-run absolute performance (grid cells)")
    lines.append("=" * 60)
    lines.append(
        "Each section is independent: IQM / mean return are from that run’s checkpoint only."
    )
    lines.append("")

    for label in sorted(grids.keys()):
        g = grids[label]
        iqm = [float(r["iqm_return"]) for r in g.rows]
        mean_r = [float(r["mean_return"]) for r in g.rows]
        si = scalar_stats(iqm)
        sm = scalar_stats(mean_r)

        folder = folder_by_label.get(label, "")
        bline = read_baseline_iqm(root, folder) if folder else None

        lines.append(f"--- {label} ---")
        lines.append(f"  grid: {g.path}")
        if bline is not None:
            lines.append(f"  baseline_iqm (sigma=0, from baseline_summary.txt): {bline:.6f}")
        lines.append(
            f"  iqm_return over {int(si['n'])} cells: "
            f"mean={si['mean']:.6f}  median={si['median']:.6f}  stdev={si['stdev']:.6f}"
        )
        lines.append(
            f"    min={si['min']:.6f}  p90={si['p90']:.6f}  p95={si['p95']:.6f}  max={si['max']:.6f}"
        )
        if args.bootstrap > 0:
            lo, hi = bootstrap_mean_ci(iqm, n_boot=args.bootstrap)
            lines.append(f"    mean iqm 95% bootstrap CI (over cells): [{lo:.6f}, {hi:.6f}]")
        lines.append(
            f"  mean_return over cells: mean={sm['mean']:.6f}  median={sm['median']:.6f}  "
            f"min={sm['min']:.6f}  max={sm['max']:.6f}"
        )
        lines.append("")

    report = "\n".join(lines)
    print(report)
    out_path = os.path.join(root, "q4_drift_own_performance_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
