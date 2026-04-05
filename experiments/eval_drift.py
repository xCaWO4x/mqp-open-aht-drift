"""
Evaluate a trained GPL checkpoint under OU-process composition drift.

Loads a trained GPLAgent, wraps LBF with DriftWrapper at specified
(sigma, theta) parameters, and records per-episode returns, compositions,
and OU states. Computes IQM return and degradation relative to the
stationary baseline (sigma=0 row in the sweep grid).

Usage:
    # Full sweep (include sigma=0 in grid for baseline):
    python experiments/eval_drift.py \\
        --checkpoint results/gpl_lbf_train_01_paper_full/checkpoints/gpl_final.pt \\
        --sweep

    # Single point:
    python experiments/eval_drift.py --checkpoint path.pt --sigma 0.2 --theta 0.15

    # Smoke test:
    python experiments/eval_drift.py --checkpoint path.pt --sweep --smoke-test
"""

import argparse
import os
import sys
import time
import itertools

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from lbforaging.foraging.environment import ForagingEnv

from agents.gpl.gpl_agent import GPLAgent
from drift.ou_process import OUProcess
from envs.drift_wrapper import DriftWrapper
from envs.env_utils import preprocess_lbf


# ======================================================================
# Environment creation (same as train_gpl.py)
# ======================================================================

def make_lbf_env(cfg: dict, seed: int = 0) -> ForagingEnv:
    """Create an LBF ForagingEnv with full level range for obs space."""
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    K = types_cfg["K"]
    n_agents = env_cfg["n_agents"]
    n_food = food_cfg["n_food"]
    grid = env_cfg["grid_size"]

    env = ForagingEnv(
        players=n_agents,
        min_player_level=np.ones(n_agents, dtype=int),
        max_player_level=np.full(n_agents, K, dtype=int),
        field_size=(grid, grid),
        max_num_food=n_food,
        min_food_level=np.ones(n_food, dtype=int),
        max_food_level=np.full(n_food, K, dtype=int),
        sight=env_cfg.get("sight", grid),
        max_episode_steps=env_cfg["max_steps"],
        force_coop=env_cfg.get("force_coop", False),
    )
    env.np_random = np.random.default_rng(seed)
    return env


# ======================================================================
# IQM computation
# ======================================================================

def compute_iqm(returns: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50%."""
    if len(returns) == 0:
        return 0.0
    q25, q75 = np.percentile(returns, [25, 75])
    mask = (returns >= q25) & (returns <= q75)
    return float(returns[mask].mean()) if mask.sum() > 0 else float(returns.mean())


# ======================================================================
# Single (sigma, theta) evaluation
# ======================================================================

def evaluate_drift_point(
    agent: GPLAgent,
    cfg: dict,
    sigma: float,
    theta: float,
    n_episodes: int,
    seed: int,
) -> dict:
    """Evaluate a trained GPL agent under drift at a single (sigma, theta).

    Returns dict with per-episode data and aggregate metrics.
    """
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    model_cfg = cfg["model"]
    K = types_cfg["K"]
    n_agents = env_cfg["n_agents"]
    n_food = food_cfg["n_food"]
    hidden_dim = model_cfg["hidden_dim"]
    action_dim = model_cfg["action_dim"]
    device = str(agent.device)  # match the agent's device

    food_probs = food_cfg.get("fixed_level_probs", {2: 0.6, 3: 0.4})
    food_probs = {int(k): v for k, v in food_probs.items()}
    food_mode = food_cfg.get("mode", "fixed")

    rng = np.random.default_rng(seed)

    # Create OU process and wrapped env
    ou = OUProcess(K=K, theta=theta, sigma=sigma, dt=cfg["ou"]["dt"], seed=seed)
    inner_env = make_lbf_env(cfg, seed=seed)
    env = DriftWrapper(
        inner_env, ou, n_agents=n_agents, n_food=n_food,
        food_mode=food_mode, food_level_probs=food_probs, seed=seed,
    )

    # Per-episode records
    ep_returns = []
    ep_lengths = []
    ep_compositions = []
    ep_agent_levels = []
    ep_food_levels = []
    ep_ou_states = []

    for ep in range(n_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
            obs, _ = reset_out
        else:
            obs = reset_out
        agent.reset()

        # Record episode composition
        ep_compositions.append(env.composition)
        ep_agent_levels.append(env.agent_levels)
        ep_food_levels.append(env.food_levels)
        ep_ou_states.append(env.ou_state.tolist())

        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            B, _, _ = preprocess_lbf(
                obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
            )
            B_np = B.cpu().numpy()
            # Greedy action (no exploration)
            action = agent.act(B_np, learner_idx=0, epsilon=0.0)
            # Advance hidden states (act() no longer updates them)
            agent.advance_hidden(B_np)

            # Joint action: learner at 0, teammates random
            joint_action = [rng.integers(0, action_dim) for _ in range(n_agents)]
            joint_action[0] = action

            step_out = env.step(joint_action)
            if len(step_out) == 5:
                next_obs, rewards, terminated, truncated, _ = step_out
                if isinstance(terminated, (list, tuple)):
                    done = all(terminated) or all(truncated)
                else:
                    done = bool(terminated) or bool(truncated)
            else:
                next_obs, rewards, dones_out, _ = step_out
                if isinstance(dones_out, (list, tuple)):
                    done = all(dones_out)
                else:
                    done = bool(dones_out)

            if isinstance(rewards, (list, tuple)):
                ep_return += float(rewards[0])
            else:
                ep_return += float(rewards)

            ep_len += 1
            obs = next_obs

        ep_returns.append(ep_return)
        ep_lengths.append(ep_len)

    env.close()

    returns_arr = np.array(ep_returns)

    return {
        "sigma": sigma,
        "theta": theta,
        "n_episodes": n_episodes,
        "seed": seed,
        # Aggregates
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std()),
        "iqm_return": compute_iqm(returns_arr),
        "mean_length": float(np.mean(ep_lengths)),
        # Per-episode data
        "returns": ep_returns,
        "lengths": ep_lengths,
        "compositions": ep_compositions,
        "agent_levels": ep_agent_levels,
        "food_levels": ep_food_levels,
        "ou_states": ep_ou_states,
    }


# ======================================================================
# Grid sweep
# ======================================================================

def run_sweep(
    agent: GPLAgent,
    cfg: dict,
    sigmas: list,
    thetas: list,
    n_episodes: int,
    n_seeds: int,
    base_seed: int,
    results_dir: str,
) -> dict:
    """Run full (sigma, theta) grid sweep across multiple seeds."""
    os.makedirs(results_dir, exist_ok=True)

    grid = list(itertools.product(enumerate(sigmas), enumerate(thetas)))
    # mean_returns[i, j] averaged over seeds
    mean_returns = np.zeros((len(sigmas), len(thetas)))
    iqm_returns = np.zeros((len(sigmas), len(thetas)))
    all_results = []

    print(f"Sweep: {len(sigmas)} sigmas x {len(thetas)} thetas "
          f"= {len(grid)} grid points, {n_episodes} eps x {n_seeds} seeds each")

    for (i, sigma), (j, theta) in grid:
        seed_means = []
        seed_iqms = []

        for s in range(n_seeds):
            seed = base_seed + s
            result = evaluate_drift_point(
                agent, cfg, sigma, theta, n_episodes, seed,
            )
            seed_means.append(result["mean_return"])
            seed_iqms.append(result["iqm_return"])
            all_results.append(result)

        mean_returns[i, j] = float(np.mean(seed_means))
        iqm_returns[i, j] = float(np.mean(seed_iqms))
        print(f"  sigma={sigma:.3f}, theta={theta:.3f}: "
              f"mean={mean_returns[i,j]:.4f}, iqm={iqm_returns[i,j]:.4f}")

    # --- Compute degradation relative to baseline (sigma=0 row if present) ---
    degradation = None
    baseline_iqm = None
    stability_threshold = 0.10  # 10% default

    if 0.0 in sigmas:
        baseline_idx = sigmas.index(0.0)
        # Baseline IQM: average across all theta columns at sigma=0
        # (all should be ~identical since sigma=0 → no noise)
        baseline_iqm = float(iqm_returns[baseline_idx, :].mean())
        print(f"\nBaseline IQM (sigma=0, avg over thetas): {baseline_iqm:.4f}")

        if baseline_iqm > 0:
            degradation = np.zeros_like(iqm_returns)
            for i in range(len(sigmas)):
                for j in range(len(thetas)):
                    degradation[i, j] = 1.0 - iqm_returns[i, j] / baseline_iqm
            # Stability region: grid points with < threshold degradation
            stable_mask = degradation < stability_threshold
            n_stable = int(stable_mask.sum())
            n_total = degradation.size
            print(f"Stability region ({stability_threshold:.0%} threshold): "
                  f"{n_stable}/{n_total} grid points stable")

    # --- Save CSVs ---
    csv_path = os.path.join(results_dir, "drift_eval_grid.csv")
    with open(csv_path, "w") as f:
        header = "sigma,theta,mean_return,iqm_return"
        if degradation is not None:
            header += ",degradation"
        f.write(header + "\n")
        for (i, sigma), (j, theta) in grid:
            row = f"{sigma},{theta},{mean_returns[i,j]},{iqm_returns[i,j]}"
            if degradation is not None:
                row += f",{degradation[i,j]}"
            f.write(row + "\n")
    print(f"\nGrid CSV saved to {csv_path}")

    # --- Save per-episode CSV ---
    detail_path = os.path.join(results_dir, "drift_eval_episodes.csv")
    with open(detail_path, "w") as f:
        f.write("sigma,theta,seed,episode,return,length,"
                "agent_levels,food_levels,ou_state\n")
        for r in all_results:
            for ep_idx in range(r["n_episodes"]):
                al = ";".join(str(x) for x in r["agent_levels"][ep_idx])
                fl = ";".join(str(x) for x in r["food_levels"][ep_idx])
                ou = ";".join(f"{x:.4f}" for x in r["ou_states"][ep_idx])
                f.write(f"{r['sigma']},{r['theta']},{r['seed']},{ep_idx},"
                        f"{r['returns'][ep_idx]},{r['lengths'][ep_idx]},"
                        f"{al},{fl},{ou}\n")
    print(f"Episode CSV saved to {detail_path}")

    # --- Save baseline summary ---
    if baseline_iqm is not None:
        baseline_path = os.path.join(results_dir, "baseline_summary.txt")
        with open(baseline_path, "w") as f:
            f.write(f"baseline_iqm: {baseline_iqm:.6f}\n")
            f.write(f"stability_threshold: {stability_threshold}\n")
            if degradation is not None:
                n_stable = int((degradation < stability_threshold).sum())
                f.write(f"stable_points: {n_stable}/{degradation.size}\n")
        print(f"Baseline summary saved to {baseline_path}")

    # --- Heatmaps ---
    _save_heatmap(mean_returns, iqm_returns, sigmas, thetas, results_dir)
    if degradation is not None:
        _save_degradation_heatmap(
            degradation, sigmas, thetas, baseline_iqm,
            stability_threshold, results_dir,
        )

    return {
        "mean_returns": mean_returns,
        "iqm_returns": iqm_returns,
        "degradation": degradation,
        "baseline_iqm": baseline_iqm,
        "all_results": all_results,
    }


def _save_heatmap(
    mean_returns: np.ndarray,
    iqm_returns: np.ndarray,
    sigmas: list,
    thetas: list,
    results_dir: str,
):
    """Save degradation heatmaps for mean and IQM return."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, data, title in [
            (axes[0], mean_returns, "Mean return"),
            (axes[1], iqm_returns, "IQM return"),
        ]:
            im = ax.imshow(
                data, aspect="auto", origin="lower",
                extent=[thetas[0], thetas[-1], sigmas[0], sigmas[-1]],
            )
            ax.set_xlabel("theta (mean-reversion)")
            ax.set_ylabel("sigma (noise)")
            ax.set_title(f"GPL under drift: {title}")
            fig.colorbar(im, ax=ax, label=title)

        plot_path = os.path.join(results_dir, "drift_eval_heatmap.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved to {plot_path}")
    except ImportError:
        print("[WARN] matplotlib not installed, skipping heatmap.")


def _save_degradation_heatmap(
    degradation: np.ndarray,
    sigmas: list,
    thetas: list,
    baseline_iqm: float,
    stability_threshold: float,
    results_dir: str,
):
    """Save degradation heatmap with stability boundary contour."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Diverging colormap centered at 0: blue (improvement) → white → red (degradation)
        vmax = max(abs(degradation.min()), abs(degradation.max()), 0.5)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        im = ax.imshow(
            degradation, aspect="auto", origin="lower",
            extent=[thetas[0], thetas[-1], sigmas[0], sigmas[-1]],
            cmap="RdBu_r", norm=norm,
        )
        ax.set_xlabel(r"$\theta$ (mean-reversion rate)")
        ax.set_ylabel(r"$\sigma$ (noise scale)")
        ax.set_title(
            f"GPL degradation under drift\n"
            f"(baseline IQM={baseline_iqm:.3f}, "
            f"threshold={stability_threshold:.0%})"
        )
        fig.colorbar(im, ax=ax, label="Fractional degradation (1 - IQM/baseline)")

        # Overlay stability boundary contour
        try:
            theta_arr = np.array(thetas)
            sigma_arr = np.array(sigmas)
            cs = ax.contour(
                theta_arr, sigma_arr, degradation,
                levels=[stability_threshold],
                colors="black", linewidths=2, linestyles="--",
            )
            ax.clabel(cs, fmt=f"{stability_threshold:.0%}", fontsize=10)
        except Exception:
            pass  # contour may fail with too few grid points

        # Mark stable vs degraded cells
        for i, sigma in enumerate(sigmas):
            for j, theta in enumerate(thetas):
                marker = "o" if degradation[i, j] < stability_threshold else "x"
                color = "green" if degradation[i, j] < stability_threshold else "red"
                # Map grid indices to plot coordinates
                ax.plot(theta, sigma, marker, color=color, markersize=6, alpha=0.7)

        plot_path = os.path.join(results_dir, "drift_degradation_heatmap.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Degradation heatmap saved to {plot_path}")
    except ImportError:
        print("[WARN] matplotlib not installed, skipping degradation heatmap.")


# ======================================================================
# Single-point evaluation (non-sweep mode)
# ======================================================================

def run_single(
    agent: GPLAgent,
    cfg: dict,
    sigma: float,
    theta: float,
    n_episodes: int,
    seed: int,
    results_dir: str,
) -> dict:
    """Evaluate at a single (sigma, theta) and save results."""
    os.makedirs(results_dir, exist_ok=True)

    print(f"Evaluating GPL under drift: sigma={sigma}, theta={theta}, "
          f"{n_episodes} episodes, seed={seed}")

    result = evaluate_drift_point(agent, cfg, sigma, theta, n_episodes, seed)

    print(f"  mean_return={result['mean_return']:.4f}, "
          f"iqm_return={result['iqm_return']:.4f}, "
          f"mean_length={result['mean_length']:.1f}")

    # Save per-episode CSV
    csv_path = os.path.join(
        results_dir, f"eval_s{sigma}_t{theta}_seed{seed}.csv"
    )
    with open(csv_path, "w") as f:
        f.write("episode,return,length,agent_levels,food_levels,ou_state\n")
        for ep_idx in range(n_episodes):
            al = ";".join(str(x) for x in result["agent_levels"][ep_idx])
            fl = ";".join(str(x) for x in result["food_levels"][ep_idx])
            ou = ";".join(f"{x:.4f}" for x in result["ou_states"][ep_idx])
            f.write(f"{ep_idx},{result['returns'][ep_idx]},"
                    f"{result['lengths'][ep_idx]},{al},{fl},{ou}\n")
    print(f"Results saved to {csv_path}")

    # Return trajectory plot
    _save_return_trajectory(result, results_dir)

    return result


def _save_return_trajectory(result: dict, results_dir: str):
    """Plot per-episode returns and OU state evolution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sigma = result["sigma"]
        theta = result["theta"]
        n_eps = result["n_episodes"]
        episodes = np.arange(n_eps)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Panel 1: Episode returns + running mean
        ax = axes[0]
        ax.plot(episodes, result["returns"], alpha=0.4, label="per-episode")
        if n_eps >= 10:
            window = min(20, n_eps // 3)
            running = np.convolve(
                result["returns"], np.ones(window) / window, mode="valid"
            )
            ax.plot(
                episodes[window - 1:], running,
                linewidth=2, label=f"running mean (w={window})"
            )
        ax.set_ylabel("Episode return")
        ax.set_title(f"GPL under drift: sigma={sigma}, theta={theta}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Mean agent level per episode
        ax = axes[1]
        mean_levels = [np.mean(lvls) for lvls in result["agent_levels"]]
        ax.plot(episodes, mean_levels, color="orange")
        ax.set_ylabel("Mean agent level")
        ax.grid(True, alpha=0.3)

        # Panel 3: OU state (type frequencies)
        ax = axes[2]
        ou_arr = np.array(result["ou_states"])
        K = ou_arr.shape[1]
        for k in range(K):
            ax.plot(episodes, ou_arr[:, k], label=f"type {k} (level {k+1})")
        ax.set_ylabel("Type frequency")
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(
            results_dir, f"trajectory_s{sigma}_t{theta}.png"
        )
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Trajectory plot saved to {plot_path}")
    except ImportError:
        print("[WARN] matplotlib not installed, skipping trajectory plot.")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GPL under OU composition drift"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained GPL checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", default="configs/gpl_lbf.yaml",
        help="Training config YAML (needed for model architecture)",
    )
    parser.add_argument(
        "--sweep-config", default="configs/drift_sweep.yaml",
        help="Sweep config YAML (for --sweep mode)",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.1,
        help="OU noise scale (single-point mode)",
    )
    parser.add_argument(
        "--theta", type=float, default=0.15,
        help="OU mean-reversion rate (single-point mode)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Override number of episodes",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run full (sigma, theta) grid sweep from sweep config",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick 3-episode run for verification",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override results directory",
    )
    args = parser.parse_args()

    # Load training config (for model architecture)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device: match evaluate_drift_point's logic
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Build and load agent
    model_cfg = cfg["model"]
    agent = GPLAgent(
        obs_dim=cfg["preprocess"]["obs_dim"],
        action_dim=model_cfg["action_dim"],
        type_dim=model_cfg["type_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_gnn_layers=model_cfg["n_gnn_layers"],
        pairwise_rank=model_cfg["pairwise_rank"],
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        tau=cfg.get("tau", None),
        t_update=cfg["training"]["t_update"],
        t_targ_update=cfg["training"]["t_targ_update"],
        polyak_tau=cfg["training"].get("polyak_tau", None),
        device=device,
    )
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.sweep:
        # Full grid sweep
        with open(args.sweep_config) as f:
            sweep_cfg = yaml.safe_load(f)

        sigmas = sweep_cfg["sweep"]["sigmas"]
        thetas = sweep_cfg["sweep"]["thetas"]
        n_episodes = args.n_episodes or sweep_cfg["eval"]["n_episodes"]
        n_seeds = sweep_cfg["eval"]["n_seeds"]
        results_dir = args.results_dir or "results/drift_eval"

        if args.smoke_test:
            n_episodes = 3
            n_seeds = 1
            sigmas = sigmas[:2]
            thetas = thetas[:2]

        run_sweep(
            agent, cfg, sigmas, thetas,
            n_episodes, n_seeds, args.seed, results_dir,
        )
    else:
        # Single (sigma, theta) evaluation
        n_episodes = args.n_episodes or cfg["drift_eval"]["n_episodes"]
        results_dir = args.results_dir or "results/drift_eval"

        if args.smoke_test:
            n_episodes = 3

        run_single(
            agent, cfg, args.sigma, args.theta,
            n_episodes, args.seed, results_dir,
        )


if __name__ == "__main__":
    main()
