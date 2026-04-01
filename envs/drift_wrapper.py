"""
DriftWrapper — wraps an LBF gym environment with OU-process-driven
agent-type composition drift.

On each episode reset the wrapper:
  1. Advances the OU process over the type-frequency simplex.
  2. Samples a team composition (agent levels) from the current distribution.
  3. Samples food levels according to the configured food mode.
  4. Injects both into the inner environment and resets.

Within an episode the composition and food levels are held fixed.
"""

from typing import Dict, List, Optional, Tuple

try:
    import gymnasium as gym
except ImportError:
    import gym

import numpy as np
from drift.ou_process import OUProcess


# ======================================================================
# Food-level sampling
# ======================================================================

def sample_food_levels_fixed(
    n_food: int,
    rng: np.random.Generator,
    food_level_probs: Optional[Dict[int, float]] = None,
) -> List[int]:
    """Sample food levels from a fixed distribution over {2, 3}.

    Primary setting: food distribution is independent of agent population,
    so drift induces a mismatch between agent capabilities and task.

    Parameters
    ----------
    n_food : int
        Number of food items to sample.
    rng : np.random.Generator
        Random number generator.
    food_level_probs : dict of {level: probability} or None
        Distribution over food levels. Defaults to {2: 0.6, 3: 0.4}.
    """
    if food_level_probs is None:
        food_level_probs = {2: 0.6, 3: 0.4}
    levels = list(food_level_probs.keys())
    probs = np.array([food_level_probs[l] for l in levels])
    probs = probs / probs.sum()
    return rng.choice(levels, size=n_food, p=probs).tolist()


def sample_food_levels_coupled(
    n_food: int,
    agent_levels: List[int],
    rng: np.random.Generator,
    min_level: int = 1,
    max_level: int = 3,
    concentration: float = 0.7,
) -> List[int]:
    """Sample food levels from a distribution centered on mean agent level.

    Ablation setting: food distribution is coupled to the current population,
    isolating the effect of compositional drift from task difficulty changes.

    The distribution places `concentration` probability mass on the level
    closest to the mean agent level, and spreads the remainder uniformly
    over adjacent levels.

    Parameters
    ----------
    n_food : int
    agent_levels : list of int
        Current episode agent levels.
    rng : np.random.Generator
    min_level, max_level : int
        Range of possible food levels.
    concentration : float
        Probability mass on the center level (default 0.7).
    """
    mean_level = np.mean(agent_levels)
    center = int(np.clip(np.round(mean_level), min_level, max_level))

    all_levels = list(range(min_level, max_level + 1))
    n_levels = len(all_levels)
    probs = np.full(n_levels, (1.0 - concentration) / max(n_levels - 1, 1))
    center_idx = all_levels.index(center)
    probs[center_idx] = concentration
    probs = probs / probs.sum()

    return rng.choice(all_levels, size=n_food, p=probs).tolist()


# ======================================================================
# DriftWrapper
# ======================================================================

class DriftWrapper(gym.Wrapper):
    """Gym wrapper that drifts team composition across episodes via an OU process.

    On reset:
      1. OU process advances → new type-frequency vector on the simplex.
      2. Agent levels sampled i.i.d. from this distribution (multinomial).
      3. Food levels sampled according to food_mode.
      4. Inner env's level bounds set to exact sampled levels (min=max).
      5. Inner env reset.

    The OU process types map directly to LBF agent levels:
      type 0 → level 1, type 1 → level 2, type 2 → level 3.

    Parameters
    ----------
    env : ForagingEnv
        The underlying LBF environment. Must be created with
        min_player_level=1, max_player_level=K so the observation
        space accommodates all possible levels.
    ou_process : OUProcess
        OU process instance (K types on the simplex).
    n_agents : int
        Fixed number of agents per episode.
    n_food : int
        Number of food items per episode.
    food_mode : str
        "fixed" (primary) or "coupled" (ablation).
    food_level_probs : dict or None
        For food_mode="fixed": distribution over food levels.
        Defaults to {2: 0.6, 3: 0.4}.
    food_coupled_concentration : float
        For food_mode="coupled": probability mass on center level.
    level_offset : int
        Maps type index to LBF level: level = type_index + level_offset.
        Default 1 (type 0 → level 1).
    seed : int or None
        RNG seed for food sampling.
    """

    def __init__(
        self,
        env,
        ou_process: OUProcess,
        n_agents: int,
        n_food: int = 3,
        food_mode: str = "fixed",
        food_level_probs: Optional[Dict[int, float]] = None,
        food_coupled_concentration: float = 0.7,
        level_offset: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.ou_process = ou_process
        self.n_agents = n_agents
        self.n_food = n_food
        self.food_mode = food_mode
        self.food_level_probs = food_level_probs
        self.food_coupled_concentration = food_coupled_concentration
        self.level_offset = level_offset
        self._rng = np.random.default_rng(seed)

        assert food_mode in ("fixed", "coupled"), (
            f"food_mode must be 'fixed' or 'coupled', got '{food_mode}'"
        )

        # Current episode state
        self._composition: List[int] = []       # type indices [0, K)
        self._agent_levels: List[int] = []      # LBF levels [1, K]
        self._food_levels: List[int] = []
        self._ou_state: np.ndarray = ou_process.state

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        # 1. Advance drift
        self._ou_state = self.ou_process.step()

        # 2. Sample agent composition (type indices)
        self._composition = self.ou_process.sample_composition(self.n_agents)

        # 3. Map type indices to LBF levels
        self._agent_levels = [t + self.level_offset for t in self._composition]

        # 4. Sample food levels
        if self.food_mode == "fixed":
            self._food_levels = sample_food_levels_fixed(
                self.n_food, self._rng, self.food_level_probs,
            )
        else:
            self._food_levels = sample_food_levels_coupled(
                self.n_food, self._agent_levels, self._rng,
                min_level=self.level_offset,
                max_level=self.ou_process.K - 1 + self.level_offset,
                concentration=self.food_coupled_concentration,
            )

        # 5. Inject levels into inner env
        inner = self.env.unwrapped
        agent_arr = np.array(self._agent_levels)
        food_arr = np.array(self._food_levels)
        inner.min_player_level = agent_arr
        inner.max_player_level = agent_arr
        inner.min_food_level = food_arr
        inner.max_food_level = food_arr

        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

    # ------------------------------------------------------------------
    # Properties for logging / inspection
    # ------------------------------------------------------------------

    @property
    def composition(self) -> List[int]:
        """Current team composition as type indices [0, K)."""
        return list(self._composition)

    @property
    def agent_levels(self) -> List[int]:
        """Current agent levels in LBF terms [1, K]."""
        return list(self._agent_levels)

    @property
    def food_levels(self) -> List[int]:
        """Current food levels."""
        return list(self._food_levels)

    @property
    def ou_state(self) -> np.ndarray:
        """Current OU process state (type-frequency vector on the simplex)."""
        return self._ou_state.copy()

    def episode_summary(self) -> Dict:
        """Summary statistics for the current episode (for logging).

        Returns dict with:
          target_distribution, realized_composition, agent_levels,
          food_levels, mean_agent_level, total_team_capability,
          mean_food_level.
        """
        return {
            "target_distribution": self._ou_state.tolist(),
            "realized_composition": list(self._composition),
            "agent_levels": list(self._agent_levels),
            "food_levels": list(self._food_levels),
            "mean_agent_level": float(np.mean(self._agent_levels)) if self._agent_levels else 0.0,
            "total_team_capability": int(sum(self._agent_levels)),
            "mean_food_level": float(np.mean(self._food_levels)) if self._food_levels else 0.0,
        }
