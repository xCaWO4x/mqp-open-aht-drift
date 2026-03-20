"""Unit tests for the DriftWrapper."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from drift.ou_process import OUProcess
from envs.drift_wrapper import DriftWrapper


# ---------------------------------------------------------------------------
# Minimal fake environment for testing (no LBF/Wolfpack dependency)
# ---------------------------------------------------------------------------

class FakeMultiAgentEnv(gym.Env):
    """Trivial env: obs is a zero vector, episode ends after 5 steps."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = 0.0
        done = self._step_count >= 5
        return obs, reward, done, {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDriftWrapper:
    def _make_wrapped(self, seed=42):
        inner = FakeMultiAgentEnv()
        ou = OUProcess(K=3, theta=0.15, sigma=0.3, dt=0.01, seed=seed)
        return DriftWrapper(inner, ou_process=ou, n_agents=4)

    def test_composition_changes_across_episodes(self):
        """Composition should differ between episode resets (with high probability)."""
        env = self._make_wrapped()
        compositions = []
        for _ in range(20):
            env.reset()
            compositions.append(tuple(env.composition))

        unique = set(compositions)
        # With 3 types and 4 agents, >1 unique composition in 20 resets is near-certain
        assert len(unique) > 1, "Composition never changed across 20 episodes"

    def test_composition_stable_within_episode(self):
        """Composition must not change between steps within the same episode."""
        env = self._make_wrapped()
        env.reset()
        comp_at_reset = env.composition

        for _ in range(5):
            env.step(env.action_space.sample())
            assert env.composition == comp_at_reset

    def test_gym_interface_compatibility(self):
        """Wrapper should satisfy the basic gym.Env contract."""
        env = self._make_wrapped()

        # reset returns an observation
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape

        # step returns (obs, reward, done, info)
        obs, reward, done, info = env.step(env.action_space.sample())
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_ou_state_is_on_simplex(self):
        """The exposed OU state should be a valid probability vector."""
        env = self._make_wrapped()
        for _ in range(50):
            env.reset()
            s = env.ou_state
            assert np.all(s >= 0)
            assert np.isclose(s.sum(), 1.0)

    def test_composition_length_matches_n_agents(self):
        env = self._make_wrapped()
        env.reset()
        assert len(env.composition) == 4

    def test_composition_types_in_range(self):
        env = self._make_wrapped()
        env.reset()
        assert all(0 <= t < 3 for t in env.composition)
