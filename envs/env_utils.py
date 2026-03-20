"""
Environment utilities: PREPROCESS and helpers for GPL input formatting.

Implements the PREPROCESS function from Appendix C.1 (Rahman et al. 2023):

    1. Split raw state s_t into per-agent features x_j and shared features u.
       - x_j: features whose values differ per agent (position, orientation, etc.)
       - u:   features shared across all agents (food/ball location, etc.)
    2. Concatenate: B_j = [x_j ; u] for each agent j.
    3. Return input batch B = {B_1, ..., B_N}.

This ensures each agent's type vector (computed by the LSTM) depends only
on its own trajectory + global context, not on other agents' features.

Also handles LSTM hidden state management for open agent sets:
    - New agents: initialise hidden state rows to zero.
    - Departed agents: remove corresponding hidden state rows.
    - Reordering: tracked by agent ID → index mapping.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


# ======================================================================
# PREPROCESS — Appendix C.1
# ======================================================================

def preprocess(
    raw_obs,
    agent_feature_slices: Dict[int, slice],
    shared_feature_slice: slice,
    prev_agent_ids: Optional[List[int]] = None,
    curr_agent_ids: Optional[List[int]] = None,
    prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """PREPROCESS function from Appendix C.1.

    Splits raw observation into per-agent input batch B and manages LSTM
    hidden states when the agent set changes (openness).

    Parameters
    ----------
    raw_obs : array-like
        Flat state vector s_t from the environment.
    agent_feature_slices : dict of {agent_id: slice}
        Maps each agent's ID to the slice of raw_obs containing its features.
    shared_feature_slice : slice
        Slice of raw_obs containing shared (global) features u.
    prev_agent_ids : list of int or None
        Agent IDs present at the previous timestep (for hidden state tracking).
    curr_agent_ids : list of int or None
        Agent IDs present at the current timestep.  If None, inferred from
        agent_feature_slices keys.
    prev_hidden : tuple of (h, c) each shape (N_prev, hidden_dim) or None
        LSTM hidden states from the previous timestep.
    hidden_dim : int
        LSTM hidden state dimension (for zero-initialising new agents).
    device : str
        Torch device.

    Returns
    -------
    B : Tensor, shape (N, obs_dim)
        Per-agent input batch where B[j] = [x_j ; u].
        obs_dim = agent_feature_dim + shared_feature_dim.
    hidden : tuple of (h, c) each shape (N, hidden_dim)
        LSTM hidden states aligned to the current agent ordering.
        New agents get zeros; departed agents are removed.
    curr_agent_ids : list of int
        Ordered agent IDs matching B's row ordering.
    """
    obs = np.asarray(raw_obs, dtype=np.float32)

    # Extract shared features u
    u = obs[shared_feature_slice]

    # Current agent IDs
    if curr_agent_ids is None:
        curr_agent_ids = sorted(agent_feature_slices.keys())

    # Build per-agent input batch: B_j = [x_j ; u]
    B_rows = []
    for agent_id in curr_agent_ids:
        x_j = obs[agent_feature_slices[agent_id]]
        B_rows.append(np.concatenate([x_j, u]))

    B = torch.tensor(np.stack(B_rows), dtype=torch.float32, device=device)
    N = len(curr_agent_ids)

    # --- LSTM hidden state management for openness ---
    if prev_hidden is None or prev_agent_ids is None:
        # No prior state: zero-initialise
        h = torch.zeros(N, hidden_dim, device=device)
        c = torch.zeros(N, hidden_dim, device=device)
    else:
        h_prev, c_prev = prev_hidden
        # Build a mapping from previous agent ID → row index
        prev_id_to_idx = {aid: i for i, aid in enumerate(prev_agent_ids)}

        h = torch.zeros(N, hidden_dim, device=device)
        c = torch.zeros(N, hidden_dim, device=device)

        for new_idx, agent_id in enumerate(curr_agent_ids):
            if agent_id in prev_id_to_idx:
                # Carry forward hidden state
                old_idx = prev_id_to_idx[agent_id]
                h[new_idx] = h_prev[old_idx]
                c[new_idx] = c_prev[old_idx]
            # else: new agent → stays zero-initialised

    return B, (h, c), curr_agent_ids


# ======================================================================
# Convenience wrappers for LBF and Wolfpack
# ======================================================================

def preprocess_lbf(raw_obs, n_agents: int, n_food: int = 1,
                   prev_agent_ids=None, prev_hidden=None,
                   hidden_dim=128, device="cpu"):
    """PREPROCESS specialised for Level-Based Foraging.

    LBF observations (gym-style, per-agent) are typically structured as:
        [agent_0_features, agent_1_features, ..., food_0_features, ...]

    Each agent has features like (y, x, level) and each food has (y, x, level).
    Agent features are per-agent (x_j), food features are shared (u).

    Parameters
    ----------
    raw_obs : list of arrays or single flat array
        If list: one observation per agent (multi-agent env).
        If flat: concatenated state vector.
    n_agents : int
    n_food : int
    prev_agent_ids, prev_hidden, hidden_dim, device : see preprocess()

    Returns
    -------
    B, hidden, agent_ids : see preprocess()
    """
    # LBF returns per-agent observations; we combine them
    if isinstance(raw_obs, (list, tuple)):
        # Multi-agent: each agent gets its own obs
        # For GPL, we need the global state — use the first agent's obs
        # and extract agent/food features from it.
        obs = np.asarray(raw_obs[0], dtype=np.float32)
    else:
        obs = np.asarray(raw_obs, dtype=np.float32)

    # Feature layout depends on specific LBF version.
    # Generic: assume first (n_agents * agent_feat_dim) are agent features,
    # rest are food (shared).
    # TODO: adapt to specific LBF observation space shape.
    total_features = len(obs)
    agent_feat_dim = 3  # (y, x, level) per agent — typical for LBF
    food_feat_dim = 3   # (y, x, level) per food

    agent_features_end = n_agents * agent_feat_dim

    agent_feature_slices = {}
    curr_agent_ids = list(range(n_agents))
    for i in range(n_agents):
        start = i * agent_feat_dim
        agent_feature_slices[i] = slice(start, start + agent_feat_dim)

    shared_feature_slice = slice(agent_features_end, total_features)

    return preprocess(
        obs, agent_feature_slices, shared_feature_slice,
        prev_agent_ids, curr_agent_ids, prev_hidden,
        hidden_dim, device,
    )


def preprocess_wolfpack(raw_obs, n_agents: int,
                        prev_agent_ids=None, prev_hidden=None,
                        hidden_dim=128, device="cpu"):
    """PREPROCESS specialised for Wolfpack.

    Wolfpack observations contain agent positions/orientations (per-agent)
    and prey positions (shared).

    # TODO: adapt to specific Wolfpack observation space structure.
    """
    obs = np.asarray(raw_obs, dtype=np.float32)

    # Wolfpack: agents have (x, y, orientation) = 3 features each,
    # remaining features are prey/global state.
    agent_feat_dim = 3
    agent_features_end = n_agents * agent_feat_dim

    agent_feature_slices = {}
    curr_agent_ids = list(range(n_agents))
    for i in range(n_agents):
        start = i * agent_feat_dim
        agent_feature_slices[i] = slice(start, start + agent_feat_dim)

    shared_feature_slice = slice(agent_features_end, len(obs))

    return preprocess(
        obs, agent_feature_slices, shared_feature_slice,
        prev_agent_ids, curr_agent_ids, prev_hidden,
        hidden_dim, device,
    )


# ======================================================================
# Generic helpers
# ======================================================================

def make_env(env_id: str, seed: int = 0, **kwargs):
    """Create and seed a gym environment by id."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    env = gym.make(env_id, **kwargs)
    if hasattr(env, "seed"):
        env.seed(seed)
    return env
