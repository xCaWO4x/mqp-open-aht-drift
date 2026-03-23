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
# LBF PREPROCESS
# ======================================================================

# LBF obs layout (standard gymnasium, per-agent, ego-centric):
#   Agent i's obs = [self_y, self_x, self_level,
#                    other1_y, other1_x, other1_level,
#                    ...,
#                    food1_y, food1_x, food1_level, ...]
#
# Layout sizes:
#   agent features: 3 per agent (y, x, level)
#   food features:  3 per food  (y, x, level)
#   total per-agent obs = 3 * n_agents + 3 * n_food
#
# Self is always first. Other agents follow. Then all food items.
#
# For PREPROCESS, we reconstruct global state from the multi-agent obs tuple:
#   x_j = agent j's (y, x, level) — 3 features
#   u   = food features — 3 * n_food features
#   B_j = [x_j ; u]

LBF_AGENT_FEAT_DIM = 3   # (y, x, level) per agent
LBF_FOOD_FEAT_DIM = 3    # (y, x, level) per food


def preprocess_lbf(
    raw_obs,
    n_agents: int,
    n_food: int = 1,
    prev_agent_ids: Optional[List[int]] = None,
    prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """PREPROCESS specialised for Level-Based Foraging.

    LBF (gymnasium) returns a tuple of per-agent ego-centric observations.
    Each agent's obs is: [self(3), other_agents(3 each), food(3 each)].

    We reconstruct the global state and produce:
        x_j = agent j's (y, x, level) = 3 features
        u   = food features = 3 * n_food features
        B_j = [x_j ; u]    (obs_dim = 3 + 3 * n_food)

    Parameters
    ----------
    raw_obs : tuple/list of arrays
        Per-agent observations from gymnasium LBF. Each shape (3*n_agents + 3*n_food,).
        OR a single flat array (already global state).
    n_agents : int
        Number of agents in the environment.
    n_food : int
        Number of food items.
    prev_agent_ids : list of int or None
        Agent IDs from previous timestep (for hidden state tracking).
    prev_hidden : tuple of (h, c) or None
        Previous LSTM hidden states.
    hidden_dim : int
    device : str

    Returns
    -------
    B : Tensor, shape (n_agents, 3 + 3*n_food)
    hidden : tuple of (h, c) each shape (n_agents, hidden_dim)
    agent_ids : list of int
    """
    if isinstance(raw_obs, (list, tuple)):
        # Multi-agent obs: reconstruct global state from ego-centric views.
        # From agent 0's perspective:
        #   [self(3), agent_1(3), ..., agent_{n-1}(3), food_0(3), ..., food_{m-1}(3)]
        # From agent i's perspective, self is always first, then others in order.
        #
        # Strategy: extract each agent's own features from their ego obs[0:3],
        # and shared food features from any agent's obs[n_agents*3:].

        agent_features = []   # list of (y, x, level) per agent
        for i in range(n_agents):
            obs_i = np.asarray(raw_obs[i], dtype=np.float32)
            # Self features are always the first 3 elements
            agent_features.append(obs_i[:LBF_AGENT_FEAT_DIM])

        # Food features from agent 0 (same for all agents)
        obs_0 = np.asarray(raw_obs[0], dtype=np.float32)
        food_start = n_agents * LBF_AGENT_FEAT_DIM
        food_features = obs_0[food_start:]

        # Build a flat global state: [agent_0(3), agent_1(3), ..., food(3*n_food)]
        global_state = np.concatenate(agent_features + [food_features])
    else:
        # Already a flat global state
        global_state = np.asarray(raw_obs, dtype=np.float32)
        food_start = n_agents * LBF_AGENT_FEAT_DIM

    # Build slices for the generic preprocess
    agent_feature_slices = {}
    curr_agent_ids = list(range(n_agents))
    for i in range(n_agents):
        start = i * LBF_AGENT_FEAT_DIM
        agent_feature_slices[i] = slice(start, start + LBF_AGENT_FEAT_DIM)

    shared_feature_slice = slice(n_agents * LBF_AGENT_FEAT_DIM, len(global_state))

    return preprocess(
        global_state, agent_feature_slices, shared_feature_slice,
        prev_agent_ids, curr_agent_ids, prev_hidden,
        hidden_dim, device,
    )


# ======================================================================
# Wolfpack PREPROCESS
# ======================================================================

# Wolfpack obs (Adhoc-wolfpack-v5) is a dict with keys:
#   teammate_location : (max_players * 2,) — (y, x) per player slot, -1 for inactive
#   opponent_info     : (n_prey * 6,)      — (y, x) + one-hot orientation per prey
#   oppo_actions      : (max_players - 1,) — previous teammate actions, -1 if unknown
#   num_agents        : (1,)               — number of active agents
#   remaining_flags   : (max_players,)     — 1.0 if continuing from last step, -1.0 if new/absent
#
# For PREPROCESS:
#   x_j = agent j's (y, x) from teammate_location = 2 features
#   u   = opponent_info (prey positions + orientations) = shared
#   B_j = [x_j ; u]
#
# The ego agent is included in teammate_location (it's one of the players).
# Active agents are the first `num_agents` slots.

WOLFPACK_AGENT_FEAT_DIM = 2   # (y, x) per agent


def preprocess_wolfpack(
    raw_obs,
    prev_agent_ids: Optional[List[int]] = None,
    prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """PREPROCESS specialised for Wolfpack (Adhoc-wolfpack-v5).

    Wolfpack returns a dict observation from the ad-hoc agent's perspective.
    All player positions (including ego) are in `teammate_location`.
    Prey info is in `opponent_info` (shared across all agents).

    We produce:
        x_j = agent j's (y, x) = 2 features
        u   = opponent_info (prey positions + orientations)
        B_j = [x_j ; u]    (obs_dim = 2 + len(opponent_info))

    Parameters
    ----------
    raw_obs : dict or 0-d numpy array wrapping a dict
        Wolfpack observation with keys:
        teammate_location, opponent_info, num_agents, remaining_flags, oppo_actions.
    prev_agent_ids : list of int or None
    prev_hidden : tuple of (h, c) or None
    hidden_dim : int
    device : str

    Returns
    -------
    B : Tensor, shape (n_active, 2 + opponent_info_dim)
    hidden : tuple of (h, c) each shape (n_active, hidden_dim)
    agent_ids : list of int
    """
    # Unwrap 0-d numpy array if needed
    if isinstance(raw_obs, np.ndarray) and raw_obs.ndim == 0:
        obs_dict = raw_obs.item()
    else:
        obs_dict = raw_obs

    # Number of currently active agents
    n_active = int(np.asarray(obs_dict["num_agents"])[0])

    # Extract per-agent positions from teammate_location
    teammate_loc = np.asarray(obs_dict["teammate_location"], dtype=np.float32)

    # Extract shared features (prey info)
    opponent_info = np.asarray(obs_dict["opponent_info"], dtype=np.float32)

    # Determine which agents are continuing vs new via remaining_flags
    remaining_flags = np.asarray(obs_dict["remaining_flags"], dtype=np.float32)

    # Build per-agent features: x_j = (y, x) for each active agent
    agent_features = []
    curr_agent_ids = list(range(n_active))
    for j in range(n_active):
        y = teammate_loc[2 * j]
        x = teammate_loc[2 * j + 1]
        agent_features.append(np.array([y, x], dtype=np.float32))

    # Build B_j = [x_j ; u] for each active agent
    B_rows = []
    for j in range(n_active):
        B_j = np.concatenate([agent_features[j], opponent_info])
        B_rows.append(B_j)

    B = torch.tensor(np.stack(B_rows), dtype=torch.float32, device=device)

    # --- LSTM hidden state management via remaining_flags ---
    # remaining_flags[j] == 1.0: agent j was present last step (carry hidden)
    # remaining_flags[j] != 1.0 (e.g., -1.0): new or absent (zero-init hidden)
    N = n_active
    if prev_hidden is None or prev_agent_ids is None:
        h = torch.zeros(N, hidden_dim, device=device)
        c = torch.zeros(N, hidden_dim, device=device)
    else:
        h_prev, c_prev = prev_hidden
        prev_id_to_idx = {aid: i for i, aid in enumerate(prev_agent_ids)}

        h = torch.zeros(N, hidden_dim, device=device)
        c = torch.zeros(N, hidden_dim, device=device)

        for new_idx in range(n_active):
            agent_id = curr_agent_ids[new_idx]
            # Only carry forward if flag indicates agent was present before
            if (new_idx < len(remaining_flags)
                    and remaining_flags[new_idx] == 1.0
                    and agent_id in prev_id_to_idx):
                old_idx = prev_id_to_idx[agent_id]
                if old_idx < h_prev.shape[0]:
                    h[new_idx] = h_prev[old_idx]
                    c[new_idx] = c_prev[old_idx]

    return B, (h, c), curr_agent_ids


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
