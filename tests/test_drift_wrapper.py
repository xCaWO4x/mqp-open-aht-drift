"""Unit tests for DriftWrapper with LBF level injection and food sampling."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from drift.ou_process import OUProcess
from envs.drift_wrapper import (
    DriftWrapper,
    sample_food_levels_fixed,
    sample_food_levels_coupled,
)


# ======================================================================
# Minimal fake LBF-like environment for unit tests
# ======================================================================

class FakeLBFEnv(gym.Env):
    """Minimal env that mimics ForagingEnv's level-injection interface.

    Has min/max_player_level and min/max_food_level arrays that DriftWrapper
    can write to, plus an unwrapped property pointing to self.
    """

    def __init__(self, n_agents=3, n_food=3, K=3):
        super().__init__()
        self.n_agents = n_agents
        self.n_food = n_food
        self.min_player_level = np.ones(n_agents, dtype=int)
        self.max_player_level = np.full(n_agents, K, dtype=int)
        self.min_food_level = np.ones(n_food, dtype=int)
        self.max_food_level = np.full(n_food, K, dtype=int)
        obs_dim = n_agents * 3 + n_food * 3  # LBF-like
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)
        self._step_count = 0
        # Track what levels were set at reset for verification
        self._last_player_levels = None
        self._last_food_levels = None

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        self._step_count = 0
        self._last_player_levels = self.min_player_level.copy()
        self._last_food_levels = self.min_food_level.copy()
        return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return obs, 0.0, self._step_count >= 5, {}


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def fake_env():
    return FakeLBFEnv(n_agents=3, n_food=3, K=3)


@pytest.fixture
def ou_process():
    return OUProcess(K=3, theta=0.15, sigma=0.3, dt=0.01, seed=42)


@pytest.fixture
def wrapper_fixed(fake_env, ou_process):
    return DriftWrapper(
        fake_env, ou_process, n_agents=3, n_food=3,
        food_mode="fixed", seed=42,
    )


@pytest.fixture
def wrapper_coupled(fake_env, ou_process):
    return DriftWrapper(
        fake_env, ou_process, n_agents=3, n_food=3,
        food_mode="coupled", seed=42,
    )


# ======================================================================
# Food sampling functions
# ======================================================================

class TestFoodSamplingFixed:

    def test_returns_correct_count(self):
        rng = np.random.default_rng(0)
        levels = sample_food_levels_fixed(5, rng)
        assert len(levels) == 5

    def test_levels_in_expected_range(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            levels = sample_food_levels_fixed(3, rng)
            assert all(l in (2, 3) for l in levels)

    def test_custom_distribution(self):
        rng = np.random.default_rng(0)
        # All level 2
        levels = sample_food_levels_fixed(10, rng, {2: 1.0, 3: 0.0})
        assert all(l == 2 for l in levels)

    def test_respects_probabilities(self):
        """Over many samples, distribution should match specified probs."""
        rng = np.random.default_rng(42)
        levels = sample_food_levels_fixed(10000, rng, {2: 0.7, 3: 0.3})
        frac_2 = levels.count(2) / len(levels) if isinstance(levels, list) else sum(l == 2 for l in levels) / len(levels)
        assert abs(frac_2 - 0.7) < 0.03


class TestFoodSamplingCoupled:

    def test_returns_correct_count(self):
        rng = np.random.default_rng(0)
        levels = sample_food_levels_coupled(5, [1, 2, 3], rng)
        assert len(levels) == 5

    def test_levels_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            levels = sample_food_levels_coupled(3, [1, 2, 3], rng, min_level=1, max_level=3)
            assert all(1 <= l <= 3 for l in levels)

    def test_centered_on_mean(self):
        """With high concentration, most food should be near mean agent level."""
        rng = np.random.default_rng(42)
        # All agents level 1 → food should mostly be level 1
        levels = sample_food_levels_coupled(
            1000, [1, 1, 1], rng, concentration=0.9,
        )
        frac_1 = sum(l == 1 for l in levels) / len(levels)
        assert frac_1 > 0.8

    def test_different_agent_levels_shift_center(self):
        """Different agent compositions should produce different food distributions."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        food_low = sample_food_levels_coupled(1000, [1, 1, 1], rng1, concentration=0.9)
        food_high = sample_food_levels_coupled(1000, [3, 3, 3], rng2, concentration=0.9)
        mean_low = np.mean(food_low)
        mean_high = np.mean(food_high)
        assert mean_high > mean_low + 0.5


# ======================================================================
# DriftWrapper — composition and level injection
# ======================================================================

class TestDriftWrapperComposition:

    def test_composition_changes_across_episodes(self, wrapper_fixed):
        compositions = []
        for _ in range(20):
            wrapper_fixed.reset()
            compositions.append(tuple(wrapper_fixed.composition))
        assert len(set(compositions)) > 1

    def test_composition_stable_within_episode(self, wrapper_fixed):
        wrapper_fixed.reset()
        comp = wrapper_fixed.composition
        for _ in range(5):
            wrapper_fixed.step(wrapper_fixed.action_space.sample())
            assert wrapper_fixed.composition == comp

    def test_composition_length(self, wrapper_fixed):
        wrapper_fixed.reset()
        assert len(wrapper_fixed.composition) == 3

    def test_composition_types_in_range(self, wrapper_fixed):
        wrapper_fixed.reset()
        assert all(0 <= t < 3 for t in wrapper_fixed.composition)

    def test_agent_levels_match_composition(self, wrapper_fixed):
        """agent_levels should be composition + level_offset."""
        wrapper_fixed.reset()
        for t, l in zip(wrapper_fixed.composition, wrapper_fixed.agent_levels):
            assert l == t + 1  # level_offset=1

    def test_levels_injected_into_env(self, wrapper_fixed):
        """Inner env's level arrays should be set to exact sampled levels."""
        wrapper_fixed.reset()
        inner = wrapper_fixed.env.unwrapped
        expected = np.array(wrapper_fixed.agent_levels)
        np.testing.assert_array_equal(inner._last_player_levels, expected)

    def test_food_levels_injected_into_env(self, wrapper_fixed):
        wrapper_fixed.reset()
        inner = wrapper_fixed.env.unwrapped
        expected = np.array(wrapper_fixed.food_levels)
        np.testing.assert_array_equal(inner._last_food_levels, expected)


# ======================================================================
# DriftWrapper — OU state
# ======================================================================

class TestDriftWrapperOUState:

    def test_ou_state_on_simplex(self, wrapper_fixed):
        for _ in range(50):
            wrapper_fixed.reset()
            s = wrapper_fixed.ou_state
            assert np.all(s >= 0)
            assert np.isclose(s.sum(), 1.0)

    def test_gym_interface(self, wrapper_fixed):
        obs = wrapper_fixed.reset()
        assert isinstance(obs, np.ndarray)
        obs, reward, done, info = wrapper_fixed.step(wrapper_fixed.action_space.sample())
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)


# ======================================================================
# DriftWrapper — food modes
# ======================================================================

class TestDriftWrapperFoodModes:

    def test_fixed_mode_food_levels(self, wrapper_fixed):
        """In fixed mode, food levels should be from {2, 3}."""
        for _ in range(20):
            wrapper_fixed.reset()
            assert all(l in (2, 3) for l in wrapper_fixed.food_levels)

    def test_coupled_mode_food_levels(self, wrapper_coupled):
        """In coupled mode, food levels should be in [1, 3]."""
        for _ in range(20):
            wrapper_coupled.reset()
            assert all(1 <= l <= 3 for l in wrapper_coupled.food_levels)

    def test_food_count(self, wrapper_fixed):
        wrapper_fixed.reset()
        assert len(wrapper_fixed.food_levels) == 3

    def test_invalid_food_mode(self, fake_env, ou_process):
        with pytest.raises(AssertionError, match="food_mode"):
            DriftWrapper(fake_env, ou_process, n_agents=3, food_mode="invalid")


# ======================================================================
# DriftWrapper — episode_summary
# ======================================================================

class TestDriftWrapperSummary:

    def test_episode_summary_keys(self, wrapper_fixed):
        wrapper_fixed.reset()
        summary = wrapper_fixed.episode_summary()
        expected_keys = {
            "target_distribution", "realized_composition", "agent_levels",
            "food_levels", "mean_agent_level", "total_team_capability",
            "mean_food_level",
        }
        assert set(summary.keys()) == expected_keys

    def test_summary_values_consistent(self, wrapper_fixed):
        wrapper_fixed.reset()
        s = wrapper_fixed.episode_summary()
        assert len(s["realized_composition"]) == 3
        assert len(s["agent_levels"]) == 3
        assert len(s["food_levels"]) == 3
        assert s["total_team_capability"] == sum(s["agent_levels"])
        assert abs(s["mean_agent_level"] - np.mean(s["agent_levels"])) < 1e-6
        assert abs(s["mean_food_level"] - np.mean(s["food_levels"])) < 1e-6
        assert len(s["target_distribution"]) == 3
        assert abs(sum(s["target_distribution"]) - 1.0) < 1e-6
