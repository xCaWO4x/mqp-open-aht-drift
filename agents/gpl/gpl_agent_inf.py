"""
GPL agent with auxiliary inference head and EMA belief context.

Extends GPLAgent with two additions:
  1. AuxiliaryLevelHead: predicts teammate levels from type embeddings,
     trained with cross-entropy against privileged labels (CTDE).
  2. EMA context: a fixed-size vector summarizing recent type embeddings,
     concatenated to the observation input so the policy can condition
     on population-level trends.

The obs_dim of the underlying GPLAgent is increased by ema_dim to
accommodate the concatenated EMA context. The auxiliary head attaches
to the type_net_agent's output and adds its loss to the agent model
optimizer.

Used by Q3-inf / Q4-inf experiments. Q3 / Q4 (without -inf) use the
plain GPLAgent as baselines.
"""

import numpy as np
import torch
import torch.nn as nn

from agents.gpl.gpl_agent import GPLAgent
from agents.gpl.auxiliary_head import AuxiliaryLevelHead
from drift.ema_tracker import EMABeliefTracker


class GPLAgentInf(GPLAgent):
    """GPL agent with auxiliary inference and EMA belief tracking.

    Parameters
    ----------
    obs_dim : int
        Base observation dimension (before EMA concatenation).
    ema_dim : int
        Dimension of the EMA context vector. The effective obs_dim
        passed to the parent GPLAgent is obs_dim + ema_dim.
    aux_n_classes : int
        Number of level classes for the auxiliary head (default 3).
    aux_weight : float
        Weight of the auxiliary loss relative to the agent model loss.
    ema_alpha : float
        EMA decay rate (higher = more weight on recent).
    **kwargs
        All other GPLAgent parameters (action_dim, type_dim, hidden_dim, etc.).
    """

    def __init__(
        self,
        obs_dim: int,
        ema_dim: int = 0,
        aux_n_classes: int = 3,
        aux_weight: float = 0.1,
        ema_alpha: float = 0.1,
        **kwargs,
    ):
        self._base_obs_dim = obs_dim
        self._ema_dim = ema_dim
        effective_obs_dim = obs_dim + ema_dim

        # Initialize parent with expanded obs_dim
        super().__init__(obs_dim=effective_obs_dim, **kwargs)

        # --- Auxiliary head ---
        type_dim = kwargs.get("type_dim", 32)
        self.aux_head = AuxiliaryLevelHead(
            type_dim=type_dim,
            n_classes=aux_n_classes,
        ).to(self.device)
        self.aux_weight = aux_weight

        # Add aux head parameters to agent model optimizer
        self.optimiser_agent.add_param_group(
            {"params": self.aux_head.parameters()}
        )

        # --- EMA belief tracker ---
        self.ema_tracker = EMABeliefTracker(dim=ema_dim, alpha=ema_alpha)

        # Accumulator for type embeddings within an episode (for EMA update)
        self._episode_type_embs = []

    def augment_obs(self, B_np: np.ndarray) -> np.ndarray:
        """Concatenate EMA context to observation batch.

        Parameters
        ----------
        B_np : np.ndarray, shape (N, base_obs_dim)

        Returns
        -------
        B_aug : np.ndarray, shape (N, base_obs_dim + ema_dim)
        """
        if self._ema_dim == 0:
            return B_np
        N = B_np.shape[0]
        ema = self.ema_tracker.context  # (ema_dim,)
        ema_tiled = np.tile(ema, (N, 1))  # (N, ema_dim)
        return np.concatenate([B_np, ema_tiled], axis=1)

    def train_step_online_inf(
        self,
        B_t_raw: np.ndarray,
        joint_actions,
        reward: float,
        B_t_next_raw: np.ndarray,
        done: bool,
        agent_levels: list,
        learner_idx: int = 0,
        teammate_indices=None,
    ):
        """Extended training step with auxiliary loss.

        Same as GPLAgent.train_step_online but additionally:
        1. Augments observations with EMA context
        2. Computes auxiliary level prediction loss on type embeddings
        3. Accumulates type embeddings for end-of-episode EMA update

        Parameters
        ----------
        B_t_raw : np.ndarray, shape (N, base_obs_dim)
            Raw preprocessed observation (without EMA).
        joint_actions : array-like, shape (N,)
        reward : float
        B_t_next_raw : np.ndarray, shape (N, base_obs_dim)
        done : bool
        agent_levels : list of int
            Privileged ground-truth LBF levels for each agent (1-indexed).
            Used only for the auxiliary loss — NOT fed to the policy.
        learner_idx : int
        teammate_indices : list or None

        Returns
        -------
        metrics : dict or None
        """
        # Augment with EMA
        B_t = self.augment_obs(B_t_raw)
        B_t_next = self.augment_obs(B_t_next_raw)

        # Snapshot the pre-update hidden state so the aux-loss forward
        # sees the same temporal context the policy itself uses. If we
        # instead passed None (no hidden), the LSTM would see a single
        # frame with no history — under observe_agent_levels=false the
        # level label is NOT recoverable from a single frame, so the aux
        # CE would stay pinned at uniform (ln K ≈ 1.10 for K=3) and the
        # head never learns. This was the prior behavior.
        hidden_agent_prev = self._hidden_agent

        # --- Standard GPL training step (parent) ---
        metrics = super().train_step_online(
            B_t, joint_actions, reward, B_t_next, done,
            learner_idx=learner_idx,
            teammate_indices=teammate_indices,
        )

        # --- Auxiliary loss: predict teammate levels from type embeddings ---
        # Fresh forward (independent graph for aux backward) but with the
        # pre-update carried hidden state — matches what the policy saw.
        B_t_tensor = torch.FloatTensor(B_t).to(self.device)
        type_emb, _ = self.type_net_agent(B_t_tensor, hidden_agent_prev)

        levels_tensor = torch.LongTensor(agent_levels).to(self.device)

        N = len(agent_levels)
        if teammate_indices is None:
            teammate_indices = [j for j in range(N) if j != learner_idx]

        if len(teammate_indices) > 0 and self.aux_weight > 0:
            tm_emb = type_emb[teammate_indices]
            tm_levels = levels_tensor[teammate_indices]
            aux_loss = self.aux_head.loss(tm_emb, tm_levels) * self.aux_weight

            (aux_loss / self.t_update).backward()

            if metrics is not None:
                metrics["aux_loss"] = aux_loss.item()

        # Accumulate type embeddings for EMA
        with torch.no_grad():
            mean_emb = type_emb.detach().mean(dim=0).cpu().numpy()
            self._episode_type_embs.append(mean_emb)

        # On episode end, update EMA tracker
        if done and len(self._episode_type_embs) > 0:
            episode_mean = np.mean(self._episode_type_embs, axis=0)
            # Project to ema_dim if type_dim != ema_dim
            if self._ema_dim > 0:
                # Use first ema_dim dimensions (or pad if needed)
                if len(episode_mean) >= self._ema_dim:
                    self.ema_tracker.update(episode_mean[:self._ema_dim])
                else:
                    padded = np.zeros(self._ema_dim)
                    padded[:len(episode_mean)] = episode_mean
                    self.ema_tracker.update(padded)
            self._episode_type_embs = []

        return metrics

    def act_inf(self, B_t_raw: np.ndarray, learner_idx=0, epsilon=0.0):
        """Action selection with EMA-augmented observation.

        Parameters
        ----------
        B_t_raw : np.ndarray, shape (N, base_obs_dim)
        """
        B_t = self.augment_obs(B_t_raw)
        return self.act(B_t, learner_idx=learner_idx, epsilon=epsilon)

    def advance_hidden_inf(self, B_t_raw: np.ndarray):
        """Advance hidden states with EMA-augmented observation."""
        B_t = self.augment_obs(B_t_raw)
        self.advance_hidden(B_t)

        # Accumulate for EMA update during eval
        with torch.no_grad():
            B_t_tensor = torch.FloatTensor(B_t).to(self.device)
            type_emb, _ = self.type_net_agent(B_t_tensor, None)
            mean_emb = type_emb.mean(dim=0).cpu().numpy()
            self._episode_type_embs.append(mean_emb)

    def end_episode_ema(self):
        """Finalize EMA update at end of episode (call during eval)."""
        if len(self._episode_type_embs) > 0:
            episode_mean = np.mean(self._episode_type_embs, axis=0)
            if self._ema_dim > 0 and len(episode_mean) >= self._ema_dim:
                self.ema_tracker.update(episode_mean[:self._ema_dim])
            self._episode_type_embs = []

    # ------------------------------------------------------------------
    # Persistence (extended)
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save all model weights including aux head and EMA state."""
        base_state = {
            "type_net_q": self.type_net_q.state_dict(),
            "type_net_agent": self.type_net_agent.state_dict(),
            "agent_model": self.agent_model.state_dict(),
            "q_network": self.q_network.state_dict(),
            "q_network_target": self.q_network_target.state_dict(),
            "type_net_q_target": self.type_net_q_target.state_dict(),
            "optimiser_q": self.optimiser_q.state_dict(),
            "optimiser_agent": self.optimiser_agent.state_dict(),
            "step_count": self._step_count,
            "aux_head": self.aux_head.state_dict(),
            "ema_tracker": self.ema_tracker.state_dict(),
        }
        torch.save(base_state, path)

    def load(self, path: str):
        """Load model weights including aux head and EMA state."""
        try:
            ckpt = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.type_net_q.load_state_dict(ckpt["type_net_q"])
        self.type_net_agent.load_state_dict(ckpt["type_net_agent"])
        self.agent_model.load_state_dict(ckpt["agent_model"])
        self.q_network.load_state_dict(ckpt["q_network"])
        self.q_network_target.load_state_dict(ckpt["q_network_target"])
        self.type_net_q_target.load_state_dict(ckpt["type_net_q_target"])
        self.optimiser_q.load_state_dict(ckpt["optimiser_q"])
        self.optimiser_agent.load_state_dict(ckpt["optimiser_agent"])
        self._step_count = ckpt["step_count"]
        if "aux_head" in ckpt:
            self.aux_head.load_state_dict(ckpt["aux_head"])
        if "ema_tracker" in ckpt:
            self.ema_tracker.load_state_dict(ckpt["ema_tracker"])
