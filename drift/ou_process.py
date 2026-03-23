"""
Ornstein-Uhlenbeck process over the probability simplex.

Models smoothly drifting agent-type frequencies for open ad hoc teamwork.
The OU dynamics in unconstrained space are:

    x += theta * (mu - x) * dt + sigma * sqrt(dt) * N(0, I)

After each update, x is projected back onto the K-simplex via Euclidean
simplex projection (Duchi et al., 2008).
"""

from typing import List

import numpy as np


def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of vector v onto the probability simplex.

    Algorithm from Duchi et al. (2008) "Efficient Projections onto the
    l1-Ball for Learning in High Dimensions".
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


class OUProcess:
    """Ornstein-Uhlenbeck process over the K-simplex.

    Parameters
    ----------
    K : int
        Number of agent types (simplex dimension).
    theta : float
        Mean-reversion rate. Higher values pull state back to mu faster.
    sigma : float
        Noise scale. Higher values produce larger random perturbations.
    mu : np.ndarray or None
        Target mean on the K-simplex. Defaults to uniform (1/K, ..., 1/K).
    dt : float
        Discrete timestep size for the Euler-Maruyama update.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        K: int,
        theta: float = 0.15,
        sigma: float = 0.2,
        mu: np.ndarray = None,
        dt: float = 0.01,
        seed: int = None,
    ):
        if K < 2:
            raise ValueError("K must be >= 2")
        if theta < 0:
            raise ValueError("theta must be non-negative")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")

        self.K = K
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Target mean: uniform by default
        if mu is not None:
            mu = np.asarray(mu, dtype=float)
            if mu.shape != (K,):
                raise ValueError(f"mu must have shape ({K},), got {mu.shape}")
            if not np.isclose(mu.sum(), 1.0):
                raise ValueError("mu must sum to 1")
            self.mu = mu.copy()
        else:
            self.mu = np.ones(K) / K

        # State initialised at mu
        self.x = self.mu.copy()

    @property
    def state(self) -> np.ndarray:
        """Current type-frequency vector on the probability simplex."""
        return self.x.copy()

    def step(self) -> np.ndarray:
        """Advance the OU process by one timestep and return the new state."""
        noise = self.rng.standard_normal(self.K)
        self.x = self.x + self.theta * (self.mu - self.x) * self.dt \
                 + self.sigma * np.sqrt(self.dt) * noise
        self.x = project_onto_simplex(self.x)
        return self.state

    def reset(self) -> np.ndarray:
        """Reset state to mu plus small noise, projected onto the simplex."""
        noise = self.rng.standard_normal(self.K) * 0.01
        self.x = project_onto_simplex(self.mu + noise)
        return self.state

    def sample_composition(self, n_agents: int) -> List[int]:
        """Sample a team of n_agents by drawing types i.i.d. from current state.

        Returns a list of integer type indices in [0, K).
        """
        return self.rng.choice(self.K, size=n_agents, p=self.x).tolist()

    def __repr__(self) -> str:
        return (
            f"OUProcess(K={self.K}, theta={self.theta}, sigma={self.sigma}, "
            f"dt={self.dt}, state={self.x.round(4)})"
        )
