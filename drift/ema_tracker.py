"""
EMA belief tracker — exponential moving average of type embeddings.

Maintains a running average of inferred type vectors across episodes,
providing a "population context" signal that summarizes recent team
composition. Concatenated to policy input so the agent can condition
on population-level trends.

Purely inference-time in principle, but the policy must be trained with
the EMA visible in the input to learn to use it. During training, the
EMA is updated after each episode and included in B_t for the next episode.

Used by Q3-inf and Q4-inf experiments.
"""

import numpy as np
import torch


class EMABeliefTracker:
    """Exponential moving average of type embeddings across episodes.

    After each episode, call `update()` with the mean type embedding
    from that episode. The tracker maintains a smoothed estimate of
    the recent population composition.

    Parameters
    ----------
    dim : int
        Dimension of the type embedding.
    alpha : float
        EMA decay rate. Higher = more weight on recent episodes.
        Default 0.1 (slow adaptation).
    """

    def __init__(self, dim: int, alpha: float = 0.1):
        self.dim = dim
        self.alpha = alpha
        self._ema = np.zeros(dim, dtype=np.float32)
        self._initialized = False

    def update(self, type_emb_mean: np.ndarray):
        """Update EMA with the mean type embedding from the latest episode.

        Parameters
        ----------
        type_emb_mean : np.ndarray, shape (dim,)
            Mean type embedding across all agents and timesteps in the episode.
        """
        emb = np.asarray(type_emb_mean, dtype=np.float32).ravel()
        if not self._initialized:
            self._ema = emb.copy()
            self._initialized = True
        else:
            self._ema = (1.0 - self.alpha) * self._ema + self.alpha * emb

    @property
    def context(self) -> np.ndarray:
        """Current EMA context vector, shape (dim,)."""
        return self._ema.copy()

    @property
    def context_torch(self) -> torch.Tensor:
        """Current EMA context as a torch tensor."""
        return torch.from_numpy(self._ema.copy())

    def reset(self):
        """Reset the tracker (e.g., at the start of a new eval run)."""
        self._ema = np.zeros(self.dim, dtype=np.float32)
        self._initialized = False

    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {"ema": self._ema.copy(), "initialized": self._initialized}

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        self._ema = state["ema"].copy()
        self._initialized = state["initialized"]
