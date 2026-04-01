"""
GPL training on Level-Based Foraging.

Trains GPL under a stationary uniform composition distribution (no drift)
to establish the baseline performance. Uses Algorithm 5 (online synchronous
training) from Rahman et al. 2023.

Usage:
    python experiments/train_gpl.py
    python experiments/train_gpl.py --config configs/gpl_lbf.yaml
    python experiments/train_gpl.py --smoke-test          # 5 episodes, no logging
    python experiments/train_gpl.py --n-episodes 500      # override episode count
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from lbforaging.foraging.environment import ForagingEnv

from agents.gpl.gpl_agent import GPLAgent
from agents.baselines.random_agent import RandomAgent
from envs.env_utils import preprocess_lbf
from eval.logger import Logger


# ======================================================================
# Environment creation
# ======================================================================

def make_lbf_env(cfg: dict, seed: int = 0) -> ForagingEnv:
    """Create an LBF ForagingEnv with full level range for obs space.

    The env is created with min_level=1, max_level=K so the observation
    space can accommodate any agent/food level. Actual levels are set
    per-episode by the training loop (stationary) or DriftWrapper (drift).
    """
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
        force_coop=env_cfg.get("force_coop", True),
    )
    env.np_random = np.random.default_rng(seed)
    return env


# ======================================================================
# Stationary composition sampling (no drift — training baseline)
# ======================================================================

def sample_stationary_composition(
    n_agents: int,
    K: int,
    rng: np.random.Generator,
    mu: np.ndarray = None,
) -> list:
    """Sample agent levels from stationary distribution (uniform by default)."""
    if mu is None:
        mu = np.ones(K) / K
    types = rng.choice(K, size=n_agents, p=mu)
    return (types + 1).tolist()  # type 0 -> level 1


def sample_food_levels(n_food: int, rng: np.random.Generator, probs: dict = None) -> list:
    """Sample food levels from fixed distribution."""
    if probs is None:
        probs = {2: 0.6, 3: 0.4}
    levels = list(probs.keys())
    p = np.array([probs[l] for l in levels])
    p = p / p.sum()
    return rng.choice(levels, size=n_food, p=p).tolist()


def inject_levels(env: ForagingEnv, agent_levels: list, food_levels: list):
    """Set exact agent and food levels on the env before reset."""
    env.min_player_level = np.array(agent_levels)
    env.max_player_level = np.array(agent_levels)
    env.min_food_level = np.array(food_levels)
    env.max_food_level = np.array(food_levels)


# ======================================================================
# Evaluation
# ======================================================================

def evaluate(
    agent: GPLAgent,
    env: ForagingEnv,
    n_episodes: int,
    n_agents: int,
    n_food: int,
    K: int,
    rng: np.random.Generator,
    food_probs: dict = None,
    hidden_dim: int = 100,
    device: str = "cpu",
) -> dict:
    """Evaluate agent for n_episodes under stationary composition, no training."""
    returns = []
    lengths = []

    for _ in range(n_episodes):
        agent_levels = sample_stationary_composition(n_agents, K, rng)
        food_levels = sample_food_levels(n_food, rng, food_probs)
        inject_levels(env, agent_levels, food_levels)

        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
            obs, _ = reset_out
        else:
            obs = reset_out
        agent.reset()

        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            B, _, _ = preprocess_lbf(
                obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
            )
            action = agent.act(B.numpy(), learner_idx=0, epsilon=0.0)

            # All agents act: learner picks action, others use uniform random
            joint_action = [rng.integers(0, 6) for _ in range(n_agents)]
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

        returns.append(ep_return)
        lengths.append(ep_len)

    returns = np.array(returns)
    # IQM: mean of the middle 50%
    q25, q75 = np.percentile(returns, [25, 75])
    mask = (returns >= q25) & (returns <= q75)
    iqm = returns[mask].mean() if mask.sum() > 0 else returns.mean()

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "iqm_return": float(iqm),
        "mean_length": float(np.mean(lengths)),
    }


# ======================================================================
# Training loop — Algorithm 5 (online synchronous)
# ======================================================================

def train(cfg: dict, smoke_test: bool = False):
    """Main training loop."""
    # --- Config ---
    env_cfg = cfg["env"]
    types_cfg = cfg["types"]
    food_cfg = cfg["food"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eps_cfg = cfg["epsilon"]
    eval_cfg = cfg["eval"]
    log_cfg = cfg["logging"]

    n_agents = env_cfg["n_agents"]
    n_food = food_cfg["n_food"]
    K = types_cfg["K"]
    obs_dim = cfg["preprocess"]["obs_dim"]
    action_dim = model_cfg["action_dim"]
    hidden_dim = model_cfg["hidden_dim"]
    n_episodes = 5 if smoke_test else train_cfg["n_episodes"]
    device = "cpu"
    seed = cfg.get("seed", 42)

    food_probs = food_cfg.get("fixed_level_probs", {2: 0.6, 3: 0.4})
    # Convert YAML string keys to int if needed
    food_probs = {int(k): v for k, v in food_probs.items()}

    # --- RNG ---
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # --- Environment ---
    env = make_lbf_env(cfg, seed=seed)

    # --- Agent ---
    agent = GPLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        type_dim=model_cfg["type_dim"],
        hidden_dim=hidden_dim,
        n_gnn_layers=model_cfg["n_gnn_layers"],
        pairwise_rank=model_cfg["pairwise_rank"],
        lr=train_cfg["lr"],
        gamma=train_cfg["gamma"],
        tau=cfg.get("tau", None),
        t_update=train_cfg["t_update"],
        t_targ_update=train_cfg["t_targ_update"],
        device=device,
    )

    # --- Logger ---
    logger = None
    if not smoke_test:
        logger = Logger(
            log_dir=log_cfg.get("log_dir", "runs/gpl_lbf"),
            use_wandb=False,  # user can init wandb externally
            use_tensorboard=log_cfg.get("tensorboard", True),
        )

    # --- Checkpoint dir ---
    results_dir = log_cfg.get("results_dir", "results/gpl_lbf")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Epsilon schedule ---
    eps_init = eps_cfg["init"]
    eps_final = eps_cfg["final"]
    eps_decay = eps_cfg["decay_episodes"]

    def get_epsilon(episode: int) -> float:
        frac = min(episode / max(eps_decay, 1), 1.0)
        return eps_init + (eps_final - eps_init) * frac

    # --- Training ---
    print(f"Training GPL on LBF: {n_episodes} episodes, "
          f"K={K}, n_agents={n_agents}, n_food={n_food}, obs_dim={obs_dim}")
    print(f"Config: lr={train_cfg['lr']}, gamma={train_cfg['gamma']}, "
          f"t_update={train_cfg['t_update']}, t_targ_update={train_cfg['t_targ_update']}")

    all_returns = []
    t_start = time.time()

    for episode in range(n_episodes):
        epsilon = get_epsilon(episode)

        # Sample stationary composition for this episode
        agent_levels = sample_stationary_composition(n_agents, K, rng)
        food_levels = sample_food_levels(n_food, rng, food_probs)
        inject_levels(env, agent_levels, food_levels)

        reset_out = env.reset()
        # Gymnasium returns (obs, info); gym returns just obs
        if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
            obs, _ = reset_out
        else:
            obs = reset_out
        agent.reset()

        ep_return = 0.0
        ep_len = 0
        done = False
        last_metrics = None

        while not done:
            # PREPROCESS: raw obs -> B_t
            B, _, _ = preprocess_lbf(
                obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
            )
            B_np = B.numpy()

            # Select action (epsilon-greedy)
            action = agent.act(B_np, learner_idx=0, epsilon=epsilon)

            # Build joint action: learner at index 0, teammates random
            joint_action = [rng.integers(0, action_dim) for _ in range(n_agents)]
            joint_action[0] = action

            # Step environment
            step_out = env.step(joint_action)
            # Gymnasium step returns (obs, reward, terminated, truncated, info)
            # Gym step returns (obs, reward, done, info)
            if len(step_out) == 5:
                next_obs, rewards, terminated, truncated, infos = step_out
                # Per-agent terminated/truncated
                if isinstance(terminated, (list, tuple)):
                    done = all(terminated) or all(truncated)
                else:
                    done = bool(terminated) or bool(truncated)
            else:
                next_obs, rewards, dones, infos = step_out
                if isinstance(dones, (list, tuple)):
                    done = all(dones)
                else:
                    done = bool(dones)

            # Per-agent rewards → learner reward
            if isinstance(rewards, (list, tuple)):
                reward = float(rewards[0])
            else:
                reward = float(rewards)

            # PREPROCESS next state
            B_next, _, _ = preprocess_lbf(
                next_obs, n_agents, n_food,
                hidden_dim=hidden_dim, device=device,
            )
            B_next_np = B_next.numpy()

            # Build joint action array for training
            joint_action_arr = np.array(joint_action)

            # Train step (Algorithm 5)
            metrics = agent.train_step_online(
                B_np, joint_action_arr, reward, B_next_np, done,
                learner_idx=0,
            )
            if metrics is not None:
                last_metrics = metrics

            ep_return += reward
            ep_len += 1
            obs = next_obs

        all_returns.append(ep_return)

        # --- Logging ---
        if logger is not None:
            logger.log_episode(episode, ep_return, ep_len, {
                "epsilon": epsilon,
                "mean_agent_level": float(np.mean(agent_levels)),
            })
            if last_metrics is not None:
                logger.log_scalars("train", last_metrics, episode)

        # --- Periodic eval ---
        save_every = eval_cfg.get("save_every", 50)
        eval_every = eval_cfg.get("eval_every", 50)

        if (episode + 1) % eval_every == 0 or episode == n_episodes - 1:
            eval_result = evaluate(
                agent, env, eval_cfg.get("eval_episodes", 5),
                n_agents, n_food, K, rng, food_probs, hidden_dim, device,
            )
            recent = all_returns[-min(50, len(all_returns)):]
            elapsed = time.time() - t_start
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0

            print(f"[Ep {episode+1:4d}/{n_episodes}] "
                  f"train_avg={np.mean(recent):.3f} "
                  f"eval_iqm={eval_result['iqm_return']:.3f} "
                  f"eps={epsilon:.3f} "
                  f"eps/s={eps_per_sec:.1f}")

            if logger is not None:
                logger.log_scalars("eval", eval_result, episode)

        # --- Checkpoint ---
        if (episode + 1) % save_every == 0:
            path = os.path.join(ckpt_dir, f"gpl_ep{episode+1}.pt")
            agent.save(path)

    # Final checkpoint
    agent.save(os.path.join(ckpt_dir, "gpl_final.pt"))

    # Summary
    elapsed = time.time() - t_start
    print(f"\nTraining complete: {n_episodes} episodes in {elapsed:.1f}s")
    print(f"Final avg return (last 50): {np.mean(all_returns[-50:]):.3f}")

    if logger is not None:
        logger.close()

    return agent, all_returns


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GPL on LBF")
    parser.add_argument("--config", default="configs/gpl_lbf.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5 episodes with no logging for quick verification")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Override number of training episodes")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_episodes is not None:
        cfg["training"]["n_episodes"] = args.n_episodes
    if args.seed is not None:
        cfg["seed"] = args.seed

    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
