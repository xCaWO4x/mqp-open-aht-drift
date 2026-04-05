"""
Capability-confound analysis for the drift eval sweep.

Decomposes observed returns into contributions from team capability
(composition / level) vs. drift parameters (sigma, theta), to test
whether apparent performance changes under drift are explained by
shifts in effective team strength rather than GPL robustness.

Reads: results/drift_eval_sweep/drift_eval_episodes.csv
       (produced by eval_drift.py --sweep)

Key analyses:
  1. OLS regressions with multiple specifications
  2. Composition-categorical group averages
  3. Success-rate analysis (P(return > 0))
  4. Same-mean-level composition comparisons
  5. Partial-effect and diagnostic plots

Usage:
    python experiments/analyze_capability_confound.py \
        --episodes-csv results/drift_eval_sweep/drift_eval_episodes.csv \
        --out-dir results/capability_confound
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# Data loading
# ======================================================================

def load_episodes(csv_path: str) -> dict:
    """Load drift_eval_episodes.csv into column arrays.

    Returns dict with keys: sigma, theta, seed, episode, ret, length,
    agent_levels (list of lists), food_levels, ou_state,
    and derived: mean_level, total_level, comp_key, n_solo_capable,
    max_agent_level, min_agent_level, has_level3, success.
    """
    sigmas, thetas, seeds, episodes = [], [], [], []
    returns, lengths = [], []
    agent_levels_all, food_levels_all = [], []

    with open(csv_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            sigmas.append(float(parts[0]))
            thetas.append(float(parts[1]))
            seeds.append(int(parts[2]))
            episodes.append(int(parts[3]))
            returns.append(float(parts[4]))
            lengths.append(int(parts[5]))

            al = [int(x) for x in parts[6].split(";")]
            fl = [int(x) for x in parts[7].split(";")]
            agent_levels_all.append(al)
            food_levels_all.append(fl)

    sigma = np.array(sigmas)
    theta = np.array(thetas)
    ret = np.array(returns)
    n = len(ret)

    # Derived features
    mean_level = np.array([np.mean(al) for al in agent_levels_all])
    total_level = np.array([sum(al) for al in agent_levels_all])
    max_agent_level = np.array([max(al) for al in agent_levels_all])
    min_agent_level = np.array([min(al) for al in agent_levels_all])

    # Composition key: sorted tuple as string, e.g. "(1, 2, 3)"
    comp_key = np.array([str(tuple(sorted(al))) for al in agent_levels_all])

    # Number of agents that can solo-load at least some food (level >= 2)
    n_solo_capable = np.array([sum(1 for l in al if l >= 2)
                               for al in agent_levels_all])

    # Has at least one level-3 agent (can solo-load everything)
    has_level3 = np.array([any(l >= 3 for l in al)
                           for al in agent_levels_all], dtype=float)

    # Success: at least one food collected (return > 0)
    success = (ret > 0).astype(float)

    # Mean food level per episode
    mean_food_level = np.array([np.mean(fl) for fl in food_levels_all])

    return {
        "sigma": sigma, "theta": theta,
        "seed": np.array(seeds), "episode": np.array(episodes),
        "ret": ret, "length": np.array(lengths),
        "agent_levels": agent_levels_all,
        "food_levels": food_levels_all,
        "mean_level": mean_level, "total_level": total_level,
        "max_agent_level": max_agent_level,
        "min_agent_level": min_agent_level,
        "comp_key": comp_key,
        "n_solo_capable": n_solo_capable,
        "has_level3": has_level3,
        "success": success,
        "mean_food_level": mean_food_level,
    }


# ======================================================================
# OLS regression (numpy-only, no statsmodels dependency)
# ======================================================================

def ols_fit(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Ordinary least squares with summary statistics.

    Returns dict with coefficients, std errors, t-stats, R^2, etc.
    """
    n, p = X.shape
    # Add intercept
    X_aug = np.column_stack([np.ones(n), X])
    names = ["intercept"] + list(feature_names)

    # beta = (X'X)^{-1} X'y
    XtX = X_aug.T @ X_aug
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ (X_aug.T @ y)
    y_hat = X_aug @ beta
    resid = y - y_hat

    dof = n - (p + 1)
    sse = float(resid @ resid)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(dof, 1)

    mse = sse / max(dof, 1)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / np.where(se > 0, se, 1e-12)

    return {
        "names": names,
        "beta": beta,
        "se": se,
        "t": t_stats,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "p": p,
        "resid": resid,
        "y_hat": y_hat,
    }


def format_ols(result: dict, title: str) -> str:
    """Format OLS result as a readable table."""
    lines = [
        f"\n{'='*60}",
        f"  {title}",
        f"  n={result['n']}, R²={result['r2']:.4f}, adj-R²={result['adj_r2']:.4f}",
        f"{'='*60}",
        f"  {'Feature':<25s} {'coef':>10s} {'se':>10s} {'t':>10s}",
        f"  {'-'*55}",
    ]
    for name, b, s, t in zip(result["names"], result["beta"],
                              result["se"], result["t"]):
        lines.append(f"  {name:<25s} {b:>10.5f} {s:>10.5f} {t:>10.2f}")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


# ======================================================================
# Composition group analysis
# ======================================================================

def composition_group_table(data: dict) -> str:
    """Group-by composition: mean return, success rate, count."""
    comp_keys = np.unique(data["comp_key"])
    rows = []
    for ck in comp_keys:
        mask = data["comp_key"] == ck
        n = int(mask.sum())
        mr = float(data["ret"][mask].mean())
        sr = float(data["success"][mask].mean())
        ml = float(data["mean_level"][mask].mean())
        tl = float(data["total_level"][mask].mean())
        iqm = _iqm(data["ret"][mask])
        rows.append((ck, n, ml, tl, mr, iqm, sr))

    rows.sort(key=lambda r: r[4])  # sort by mean return

    lines = [
        f"\n{'='*80}",
        "  Composition group statistics",
        f"{'='*80}",
        f"  {'Comp':<16s} {'N':>6s} {'mean_L':>7s} {'sum_L':>6s} "
        f"{'mean_R':>8s} {'IQM_R':>8s} {'succ%':>7s}",
        f"  {'-'*72}",
    ]
    for ck, n, ml, tl, mr, iqm, sr in rows:
        lines.append(
            f"  {ck:<16s} {n:>6d} {ml:>7.2f} {tl:>6.1f} "
            f"{mr:>8.4f} {iqm:>8.4f} {sr:>6.1%}"
        )
    lines.append(f"{'='*80}")
    return "\n".join(lines)


def same_mean_comparisons(data: dict) -> str:
    """Compare compositions with the same mean level but different structure.

    Key pairs: (1,1,3) vs (1,2,2) [mean=5/3],
               (1,2,3) vs (2,2,2) [mean=2],
               (1,3,3) vs (2,2,3) [mean=7/3].
    """
    pairs = [
        ("(1, 1, 3)", "(1, 2, 2)", "mean=1.67, sum=5"),
        ("(1, 2, 3)", "(2, 2, 2)", "mean=2.00, sum=6"),
        ("(1, 3, 3)", "(2, 2, 3)", "mean=2.33, sum=7"),
    ]
    lines = [
        f"\n{'='*80}",
        "  Same-mean-level composition comparisons",
        "  (isolates coordination structure from aggregate team strength)",
        f"{'='*80}",
    ]
    for comp_a, comp_b, desc in pairs:
        mask_a = data["comp_key"] == comp_a
        mask_b = data["comp_key"] == comp_b
        na, nb = int(mask_a.sum()), int(mask_b.sum())
        if na == 0 or nb == 0:
            lines.append(f"\n  {comp_a} vs {comp_b} ({desc}): insufficient data")
            continue
        ra, rb = data["ret"][mask_a], data["ret"][mask_b]
        sa, sb = data["success"][mask_a], data["success"][mask_b]

        lines.append(f"\n  {comp_a} vs {comp_b}  ({desc})")
        lines.append(f"  {'':>16s} {'N':>6s} {'mean_R':>8s} {'IQM_R':>8s} "
                      f"{'std_R':>8s} {'succ%':>7s}")
        lines.append(f"  {comp_a:>16s} {na:>6d} {ra.mean():>8.4f} "
                      f"{_iqm(ra):>8.4f} {ra.std():>8.4f} {sa.mean():>6.1%}")
        lines.append(f"  {comp_b:>16s} {nb:>6d} {rb.mean():>8.4f} "
                      f"{_iqm(rb):>8.4f} {rb.std():>8.4f} {sb.mean():>6.1%}")
        diff = ra.mean() - rb.mean()
        lines.append(f"  diff (A-B): {diff:>+.4f}  "
                      f"({'A better' if diff > 0 else 'B better'})")

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)


# ======================================================================
# Partial-effect analysis: capability vs drift
# ======================================================================

def partial_effect_table(data: dict) -> str:
    """Show mean return by sigma holding composition fixed, and vice versa.

    This is the core diagnostic: if returns vary more across compositions
    than across sigma values within a composition, the capability confound
    dominates the drift signal.
    """
    lines = [
        f"\n{'='*80}",
        "  Partial effects: mean return by (sigma, composition)",
        "  Rows = sigma, Columns = composition (sorted by sum_level)",
        f"{'='*80}",
    ]

    unique_sigmas = np.sort(np.unique(data["sigma"]))
    unique_comps = np.unique(data["comp_key"])
    # Sort compositions by total level
    comp_sum = {}
    for ck in unique_comps:
        mask = data["comp_key"] == ck
        comp_sum[ck] = float(data["total_level"][mask].mean())
    unique_comps = sorted(unique_comps, key=lambda c: comp_sum[c])

    # Header
    header = f"  {'sigma':>6s}"
    for ck in unique_comps:
        header += f" {ck:>12s}"
    header += f" {'row_mean':>10s}"
    lines.append(header)
    lines.append(f"  {'-' * (8 + 13 * len(unique_comps) + 11)}")

    sigma_means = []
    for sig in unique_sigmas:
        row = f"  {sig:>6.3f}"
        vals = []
        for ck in unique_comps:
            mask = (data["sigma"] == sig) & (data["comp_key"] == ck)
            n = int(mask.sum())
            if n > 0:
                v = float(data["ret"][mask].mean())
                row += f" {v:>12.4f}"
                vals.append(v)
            else:
                row += f" {'--':>12s}"
        rm = float(np.mean(vals)) if vals else 0.0
        row += f" {rm:>10.4f}"
        sigma_means.append(rm)
        lines.append(row)

    # Column means
    footer = f"  {'col_mn':>6s}"
    for ck in unique_comps:
        mask = data["comp_key"] == ck
        v = float(data["ret"][mask].mean())
        footer += f" {v:>12.4f}"
    footer += f" {np.mean(sigma_means):>10.4f}"
    lines.append(f"  {'-' * (8 + 13 * len(unique_comps) + 11)}")
    lines.append(footer)

    # Variance decomposition (simple ANOVA-style)
    grand_mean = float(data["ret"].mean())
    ss_sigma = sum(
        mask.sum() * (data["ret"][mask].mean() - grand_mean) ** 2
        for sig in unique_sigmas
        for mask in [(data["sigma"] == sig)]
    )
    ss_comp = sum(
        mask.sum() * (data["ret"][mask].mean() - grand_mean) ** 2
        for ck in unique_comps
        for mask in [(data["comp_key"] == ck)]
    )
    ss_total = float(np.sum((data["ret"] - grand_mean) ** 2))

    lines.append(f"\n  Variance decomposition (Type I SS):")
    lines.append(f"    SS_sigma / SS_total = {ss_sigma:.2f} / {ss_total:.2f} "
                 f"= {ss_sigma / ss_total:.1%}" if ss_total > 0 else
                 f"    SS_sigma / SS_total = 0 / 0")
    lines.append(f"    SS_comp  / SS_total = {ss_comp:.2f} / {ss_total:.2f} "
                 f"= {ss_comp / ss_total:.1%}" if ss_total > 0 else
                 f"    SS_comp  / SS_total = 0 / 0")
    lines.append(f"    (If SS_comp >> SS_sigma, capability confound dominates)")

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)


# ======================================================================
# Success-rate regressions
# ======================================================================

def success_rate_analysis(data: dict) -> str:
    """Linear probability model for success (return > 0)."""
    lines = [
        f"\n{'='*80}",
        "  Success-rate analysis: P(return > 0)",
        f"{'='*80}",
    ]

    # Overall success rate by sigma
    unique_sigmas = np.sort(np.unique(data["sigma"]))
    lines.append(f"\n  Success rate by sigma:")
    for sig in unique_sigmas:
        mask = data["sigma"] == sig
        sr = float(data["success"][mask].mean())
        n = int(mask.sum())
        lines.append(f"    sigma={sig:.3f}: {sr:.1%}  (n={n})")

    # Success rate by composition
    lines.append(f"\n  Success rate by composition:")
    comps = np.unique(data["comp_key"])
    comp_sr = [(ck, float(data["success"][data["comp_key"] == ck].mean()),
                int((data["comp_key"] == ck).sum()))
               for ck in comps]
    comp_sr.sort(key=lambda x: x[1])
    for ck, sr, n in comp_sr:
        lines.append(f"    {ck:<16s}: {sr:.1%}  (n={n})")

    # Linear probability model: success ~ mean_level + sigma + theta + sigma:theta
    X = np.column_stack([
        data["mean_level"],
        data["sigma"],
        data["theta"],
        data["sigma"] * data["theta"],
    ])
    result = ols_fit(X, data["success"],
                     ["mean_level", "sigma", "theta", "sigma:theta"])
    lines.append(format_ols(result,
                            "LPM: success ~ mean_level + sigma + theta + sigma:theta"))

    # With has_level3 and n_solo_capable
    X2 = np.column_stack([
        data["has_level3"],
        data["n_solo_capable"].astype(float),
        data["sigma"],
        data["theta"],
    ])
    result2 = ols_fit(X2, data["success"],
                      ["has_level3", "n_solo_capable", "sigma", "theta"])
    lines.append(format_ols(result2,
                            "LPM: success ~ has_level3 + n_solo_capable + sigma + theta"))

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)


# ======================================================================
# Helpers
# ======================================================================

def _iqm(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    q25, q75 = np.percentile(arr, [25, 75])
    mask = (arr >= q25) & (arr <= q75)
    return float(arr[mask].mean()) if mask.sum() > 0 else float(arr.mean())


# ======================================================================
# Plots
# ======================================================================

def save_plots(data: dict, out_dir: str):
    """Generate diagnostic plots for the capability confound analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    # --- Plot 1: Return vs mean_agent_level, colored by sigma ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    unique_sigmas = np.sort(np.unique(data["sigma"]))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_sigmas)))
    for sig, color in zip(unique_sigmas, cmap):
        mask = data["sigma"] == sig
        ax.scatter(data["mean_level"][mask], data["ret"][mask],
                   alpha=0.15, s=8, color=color, label=f"σ={sig:.2f}")
    ax.set_xlabel("Mean agent level")
    ax.set_ylabel("Episode return")
    ax.set_title("Return vs mean agent level\n(colored by σ)")
    ax.legend(fontsize=7, markerscale=3, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Boxplot of returns by composition ---
    ax = axes[1]
    comps = np.unique(data["comp_key"])
    comp_order = sorted(comps,
                        key=lambda c: data["total_level"][data["comp_key"] == c].mean())
    comp_returns = [data["ret"][data["comp_key"] == ck] for ck in comp_order]
    bp = ax.boxplot(comp_returns, labels=comp_order, vert=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_xlabel("Composition (sorted by total level)")
    ax.set_ylabel("Episode return")
    ax.set_title("Return distribution by composition")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 3: Mean return by sigma, per composition ---
    ax = axes[2]
    for ck in comp_order:
        means = []
        sigs_present = []
        for sig in unique_sigmas:
            mask = (data["sigma"] == sig) & (data["comp_key"] == ck)
            if mask.sum() > 0:
                means.append(float(data["ret"][mask].mean()))
                sigs_present.append(sig)
        if means:
            ax.plot(sigs_present, means, "o-", markersize=4, label=ck)
    ax.set_xlabel("σ (noise scale)")
    ax.set_ylabel("Mean return")
    ax.set_title("Mean return by σ, per composition\n(parallel lines ⇒ no interaction)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "capability_confound_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved to {path}")

    # --- Plot 4: Success rate heatmap (composition x sigma) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sr_matrix = np.full((len(comp_order), len(unique_sigmas)), np.nan)
    for i, ck in enumerate(comp_order):
        for j, sig in enumerate(unique_sigmas):
            mask = (data["comp_key"] == ck) & (data["sigma"] == sig)
            if mask.sum() > 0:
                sr_matrix[i, j] = float(data["success"][mask].mean())

    im = ax.imshow(sr_matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(unique_sigmas)))
    ax.set_xticklabels([f"{s:.2f}" for s in unique_sigmas])
    ax.set_yticks(range(len(comp_order)))
    ax.set_yticklabels(comp_order)
    ax.set_xlabel("σ (noise scale)")
    ax.set_ylabel("Composition")
    ax.set_title("Success rate by (composition, σ)")
    fig.colorbar(im, ax=ax, label="P(return > 0)")

    # Annotate cells
    for i in range(len(comp_order)):
        for j in range(len(unique_sigmas)):
            v = sr_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="black" if 0.3 < v < 0.7 else "white")

    fig.tight_layout()
    path2 = os.path.join(out_dir, "success_rate_heatmap.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Success-rate heatmap saved to {path2}")

    # --- Plot 5: Variance decomposition bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    grand_mean = data["ret"].mean()
    ss_total = float(np.sum((data["ret"] - grand_mean) ** 2))
    factors = {
        "sigma": data["sigma"],
        "theta": data["theta"],
        "composition": data["comp_key"],
        "mean_level": np.round(data["mean_level"], 2).astype(str),
        "total_level": data["total_level"].astype(str),
        "has_level3": data["has_level3"].astype(str),
    }
    shares = {}
    for fname, fvals in factors.items():
        ss = sum(
            mask.sum() * (data["ret"][mask].mean() - grand_mean) ** 2
            for level in np.unique(fvals)
            for mask in [(fvals == level)]
        )
        shares[fname] = ss / ss_total if ss_total > 0 else 0.0

    names = list(shares.keys())
    vals = [shares[n] for n in names]
    colors = ["#e74c3c" if n in ("sigma", "theta") else "#3498db" for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("SS_factor / SS_total")
    ax.set_title("Variance explained by each factor\n(blue = capability, red = drift)")
    ax.set_xlim(0, max(vals) * 1.15 if vals else 1)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    path3 = os.path.join(out_dir, "variance_decomposition.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Variance decomposition saved to {path3}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Capability-confound analysis for drift eval sweep"
    )
    parser.add_argument(
        "--episodes-csv",
        default="results/eval_drift_sweep_main/drift_eval_episodes.csv",
        help="Path to per-episode CSV from eval_drift.py",
    )
    parser.add_argument(
        "--out-dir",
        default="results/capability_confound",
        help="Output directory for tables and plots",
    )
    args = parser.parse_args()

    if not os.path.exists(args.episodes_csv):
        print(f"ERROR: {args.episodes_csv} not found. Run the drift eval sweep first.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_episodes(args.episodes_csv)
    n = len(data["ret"])
    print(f"Loaded {n} episodes from {args.episodes_csv}")

    report_sections = []

    # --- 1. Composition group table ---
    section = composition_group_table(data)
    print(section)
    report_sections.append(section)

    # --- 2. Same-mean-level comparisons ---
    section = same_mean_comparisons(data)
    print(section)
    report_sections.append(section)

    # --- 3. OLS regressions ---

    # Model A: return ~ mean_level + sigma + theta + sigma:theta
    X_a = np.column_stack([
        data["mean_level"],
        data["sigma"],
        data["theta"],
        data["sigma"] * data["theta"],
    ])
    res_a = ols_fit(X_a, data["ret"],
                    ["mean_level", "sigma", "theta", "sigma:theta"])
    section = format_ols(res_a,
                         "Model A: return ~ mean_level + sigma + theta + sigma:theta")
    print(section)
    report_sections.append(section)

    # Model B: return ~ total_level + has_level3 + sigma + theta + sigma:theta
    X_b = np.column_stack([
        data["total_level"].astype(float),
        data["has_level3"],
        data["sigma"],
        data["theta"],
        data["sigma"] * data["theta"],
    ])
    res_b = ols_fit(X_b, data["ret"],
                    ["total_level", "has_level3", "sigma", "theta", "sigma:theta"])
    section = format_ols(
        res_b,
        "Model B: return ~ total_level + has_level3 + sigma + theta + sigma:theta")
    print(section)
    report_sections.append(section)

    # Model C: return ~ comp_key_dummies + sigma + theta + sigma:theta
    # (composition as categorical — fully absorbs capability differences)
    unique_comps = np.unique(data["comp_key"])
    # Reference category: most common composition or first
    ref_comp = unique_comps[0]
    comp_dummies = []
    comp_names = []
    for ck in unique_comps[1:]:
        comp_dummies.append((data["comp_key"] == ck).astype(float))
        comp_names.append(f"comp={ck}")
    if comp_dummies:
        X_c = np.column_stack(
            comp_dummies + [
                data["sigma"],
                data["theta"],
                data["sigma"] * data["theta"],
            ]
        )
        res_c = ols_fit(X_c, data["ret"],
                        comp_names + ["sigma", "theta", "sigma:theta"])
        section = format_ols(
            res_c,
            f"Model C: return ~ comp_dummies (ref={ref_comp}) + sigma + theta + sigma:theta")
        print(section)
        report_sections.append(section)

    # Model D: return ~ sigma + theta + sigma:theta (no capability controls)
    X_d = np.column_stack([
        data["sigma"],
        data["theta"],
        data["sigma"] * data["theta"],
    ])
    res_d = ols_fit(X_d, data["ret"], ["sigma", "theta", "sigma:theta"])
    section = format_ols(res_d,
                         "Model D: return ~ sigma + theta + sigma:theta (NO capability)")
    print(section)
    report_sections.append(section)

    # R² comparison
    section = (
        f"\n{'='*60}\n"
        f"  R² comparison (capability confound test)\n"
        f"{'='*60}\n"
        f"  Model D (drift only):                   R²={res_d['r2']:.4f}\n"
        f"  Model A (+mean_level):                   R²={res_a['r2']:.4f}\n"
        f"  Model B (+total_level, has_level3):      R²={res_b['r2']:.4f}\n"
    )
    if comp_dummies:
        section += (
            f"  Model C (+composition categorical):     R²={res_c['r2']:.4f}\n"
        )
    section += (
        f"\n  If R² jumps substantially when adding capability controls,\n"
        f"  the capability confound explains more variance than drift.\n"
        f"{'='*60}"
    )
    print(section)
    report_sections.append(section)

    # --- 4. Partial-effect table ---
    section = partial_effect_table(data)
    print(section)
    report_sections.append(section)

    # --- 5. Success-rate analysis ---
    section = success_rate_analysis(data)
    print(section)
    report_sections.append(section)

    # --- 6. Plots ---
    save_plots(data, args.out_dir)

    # --- Save full report ---
    report_path = os.path.join(args.out_dir, "capability_confound_report.txt")
    with open(report_path, "w") as f:
        f.write("Capability-Confound Analysis Report\n")
        f.write(f"Source: {args.episodes_csv}\n")
        f.write(f"Episodes: {n}\n")
        f.write("\n".join(report_sections))
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
