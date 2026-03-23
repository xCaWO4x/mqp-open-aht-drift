"""
DriftWrapper — wraps any LBF or Wolfpack gym environment with OU-process-
driven agent-type composition drift.

On each episode reset the wrapper advances the OU process, samples a new
team composition, and resets the inner environment with that composition.
Within an episode the composition is held fixed.
"""

from typing import List

try:
    import gymnasium as gym
except ImportError:
    import gym

import numpy as np
from drift.ou_process import OUProcess


class DriftWrapper(gym.Wrapper):
    """Gym wrapper that drifts team composition across episodes via an OU process.

    Parameters
    ----------
    env : gym.Env
        The underlying multi-agent environment.
    ou_process : OUProcess
        OU process instance controlling the type-frequency drift.
    n_agents : int
        Number of teammate agents to sample each episode.
    """

    def __init__(self, env, ou_process: OUProcess, n_agents: int):
        super().__init__(env)
        self.ou_process = ou_process
        self.n_agents = n_agents

        # Current episode state
        self._composition: List[int] = []
        self._ou_state: np.ndarray = ou_process.state

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        # Advance the drift
        self._ou_state = self.ou_process.step()
        self._composition = self.ou_process.sample_composition(self.n_agents)

        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

    # ------------------------------------------------------------------
    # Properties for logging / inspection
    # ------------------------------------------------------------------

    @property
    def composition(self) -> List[int]:
        """Current team composition as a list of agent-type indices."""
        return list(self._composition)

    @property
    def ou_state(self) -> np.ndarray:
        """Current OU process state (type-frequency vector on the simplex)."""
        return self._ou_state.copy()
