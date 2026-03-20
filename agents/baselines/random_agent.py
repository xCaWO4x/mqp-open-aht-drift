"""Random agent baseline — uniformly samples from the action space."""

import numpy as np


class RandomAgent:
    """Sanity-check baseline that ignores all observations.

    Parameters
    ----------
    action_dim : int
        Number of discrete actions.
    seed : int or None
        Optional RNG seed for reproducibility.
    """

    def __init__(self, action_dim: int, seed: int = None):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def act(self, obs=None, **kwargs) -> int:
        return int(self.rng.integers(self.action_dim))

    def reset(self):
        pass
