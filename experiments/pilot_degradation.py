"""
Pilot degradation experiment — FIRST SCRIPT TO RUN.

Sweeps a grid of (sigma, theta) OU parameters and measures mean episodic
return under each drift regime.  Currently uses a RandomAgent as a
placeholder; GPL will be swapped in later.

Usage:
    python experiments/pilot_degradation.py                  # full sweep
    python experiments/pilot_degradation.py --dry-run        # 2 episodes, no wandb
    python experiments/pilot_degradation.py --config path    # custom config
"""

import argparse
import os
import sys
import itertools

import numpy as np
import yaml

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drift.ou_process import OUProcess
from envs.drift_wrapper import DriftWrapper
from agents.baselines.random_agent import RandomAgent


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Single grid-point evaluation
# ------------------------------------------------------------------

def evaluate_grid_point(
    sigma: float,
    theta: float,
    cfg: dict,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> list[float]:
    """Run n_episodes under a single (sigma, theta) and return episode returns."""
    env_cfg = cfg["env"]
    ou_cfg = cfg["ou"]

    ou = OUProcess(
        K=ou_cfg["K"],
        theta=theta,
        sigma=sigma,
        mu=ou_cfg.get("mu"),
        dt=ou_cfg["dt"],
        seed=seed,
    )

    # Build inner environment
    # Import lbforaging to register Foraging-* envs with gymnasium
    try:
        import lbforaging  # noqa: F401 — registers envs on import
    except ImportError:
        pass

    try:
        import gymnasium
        inner_env = gymnasium.make(env_cfg["id"])
    except ImportError:
        import gym
        inner_env = gym.make(env_cfg["id"])

    env = DriftWrapper(inner_env, ou_process=ou, n_agents=env_cfg["n_agents"])

    # TODO: Replace RandomAgent with GPLAgent once implemented.
    #       agent = GPLAgent.load("checkpoints/gpl_lbf.pt")
    n_actions = (
        inner_env.action_space[0].n
        if hasattr(inner_env.action_space, '__getitem__')
        else inner_env.action_space.n
    )
    agent = RandomAgent(action_dim=n_actions, seed=seed)

    episode_returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        agent.reset()
        ep_return = 0.0

        for step in range(max_steps):
            action = agent.act(obs)

            # Multi-agent envs expect a list of actions
            if hasattr(inner_env.action_space, '__getitem__'):
                actions = [action] + [
                    np.random.randint(n_actions)
                    for _ in range(env_cfg["n_agents"] - 1)
                ]
                result = env.step(actions)
            else:
                result = env.step(action)

            # Handle both gym (4-tuple) and gymnasium (5-tuple) returns
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            # Reward may be a list in multi-agent envs
            if isinstance(reward, (list, tuple)):
                ep_return += sum(reward)
            else:
                ep_return += reward

            if done:
                break

        episode_returns.append(ep_return)

    env.close()
    return episode_returns


# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------

def run_sweep(cfg: dict, dry_run: bool = False):
    sweep_cfg = cfg["sweep"]
    eval_cfg = cfg["eval"]
    log_cfg = cfg["logging"]

    sigmas = sweep_cfg["sigmas"]
    thetas = sweep_cfg["thetas"]
    n_episodes = 2 if dry_run else eval_cfg["n_episodes"]
    max_steps = eval_cfg["max_steps"]
    seed = eval_cfg["seed"]

    results_dir = log_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Optional wandb logging
    wandb_run = None
    if not dry_run:
        try:
            import wandb
            wandb_run = wandb.init(
                project=log_cfg["wandb_project"],
                name="pilot-degradation-sweep",
                config=cfg,
            )
        except ImportError:
            print("[WARN] wandb not installed, skipping remote logging.")

    # Results matrix: rows=sigma, cols=theta
    mean_returns = np.zeros((len(sigmas), len(thetas)))
    grid = list(itertools.product(enumerate(sigmas), enumerate(thetas)))

    print(f"Running sweep: {len(sigmas)} sigmas x {len(thetas)} thetas "
          f"= {len(grid)} grid points, {n_episodes} episodes each")

    for (i, sigma), (j, theta) in grid:
        print(f"  sigma={sigma:.3f}, theta={theta:.3f} ... ", end="", flush=True)

        ep_returns = evaluate_grid_point(
            sigma=sigma, theta=theta, cfg=cfg,
            n_episodes=n_episodes, max_steps=max_steps, seed=seed,
        )
        mean_ret = float(np.mean(ep_returns))
        mean_returns[i, j] = mean_ret
        print(f"mean_return={mean_ret:.4f}")

        if wandb_run is not None:
            wandb_run.log({
                "sigma": sigma,
                "theta": theta,
                "mean_return": mean_ret,
                "std_return": float(np.std(ep_returns)),
            })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    # CSV
    csv_path = os.path.join(results_dir, "degradation_grid.csv")
    with open(csv_path, "w") as f:
        f.write("sigma,theta,mean_return\n")
        for (i, sigma), (j, theta) in grid:
            f.write(f"{sigma},{theta},{mean_returns[i, j]}\n")
    print(f"\nResults saved to {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            mean_returns, aspect="auto", origin="lower",
            extent=[thetas[0], thetas[-1], sigmas[0], sigmas[-1]],
        )
        ax.set_xlabel("theta (mean-reversion rate)")
        ax.set_ylabel("sigma (noise scale)")
        ax.set_title("Mean episodic return under OU drift (random agent baseline)")
        fig.colorbar(im, ax=ax, label="Mean return")

        plot_path = os.path.join(results_dir, "degradation_heatmap.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

        if wandb_run is not None:
            import wandb
            wandb_run.log({"degradation_heatmap": wandb.Image(plot_path)})
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot.")

    if wandb_run is not None:
        wandb_run.finish()

    print("Done.")
    return mean_returns


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pilot degradation sweep")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "configs", "drift_sweep.yaml"),
        help="Path to sweep config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick 2-episode run with no wandb logging",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_sweep(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
