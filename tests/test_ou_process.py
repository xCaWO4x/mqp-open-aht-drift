"""Unit tests for the OU process over the simplex."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drift.ou_process import OUProcess, project_onto_simplex


class TestSimplexProjection:
    def test_already_on_simplex(self):
        v = np.array([0.2, 0.3, 0.5])
        p = project_onto_simplex(v)
        assert np.allclose(p.sum(), 1.0)
        assert np.all(p >= 0)
        np.testing.assert_allclose(p, v)

    def test_negative_entries(self):
        v = np.array([-0.5, 0.8, 1.2])
        p = project_onto_simplex(v)
        assert np.allclose(p.sum(), 1.0)
        assert np.all(p >= 0)

    def test_all_equal(self):
        v = np.array([5.0, 5.0, 5.0])
        p = project_onto_simplex(v)
        np.testing.assert_allclose(p, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)


class TestOUProcess:
    def test_state_stays_on_simplex(self):
        """State must remain on the simplex after many steps."""
        ou = OUProcess(K=5, theta=0.15, sigma=0.5, dt=0.01, seed=42)
        for _ in range(10_000):
            state = ou.step()
            assert np.all(state >= 0), f"Negative entry: {state}"
            assert np.isclose(state.sum(), 1.0), f"Sum != 1: {state.sum()}"

    def test_long_run_mean_close_to_mu(self):
        """Over many steps the time-averaged state should approach mu."""
        mu = np.array([0.1, 0.2, 0.3, 0.4])
        ou = OUProcess(K=4, theta=0.5, sigma=0.05, dt=0.01, mu=mu, seed=0)

        states = []
        # Burn-in
        for _ in range(5_000):
            ou.step()
        # Collect
        for _ in range(50_000):
            states.append(ou.step())

        empirical_mean = np.mean(states, axis=0)
        np.testing.assert_allclose(empirical_mean, mu, atol=0.05)

    def test_higher_sigma_higher_variance(self):
        """Trajectories with larger sigma should have higher variance."""
        n_steps = 20_000

        def trajectory_variance(sigma):
            ou = OUProcess(K=3, theta=0.15, sigma=sigma, dt=0.01, seed=123)
            states = [ou.step() for _ in range(n_steps)]
            return np.var(states, axis=0).mean()

        var_low = trajectory_variance(0.05)
        var_high = trajectory_variance(0.5)
        assert var_high > var_low, (
            f"Expected higher sigma to give higher variance: "
            f"var(sigma=0.5)={var_high:.6f} <= var(sigma=0.05)={var_low:.6f}"
        )

    def test_reset_near_mu(self):
        ou = OUProcess(K=3, theta=0.15, sigma=0.3, dt=0.01, seed=7)
        for _ in range(1000):
            ou.step()
        state = ou.reset()
        assert np.all(state >= 0)
        assert np.isclose(state.sum(), 1.0)
        # After reset, state should be close to mu (uniform)
        np.testing.assert_allclose(state, ou.mu, atol=0.05)

    def test_sample_composition(self):
        ou = OUProcess(K=4, theta=0.15, sigma=0.1, dt=0.01, seed=99)
        types = ou.sample_composition(10)
        assert len(types) == 10
        assert all(0 <= t < 4 for t in types)
        assert all(isinstance(t, int) for t in types)

    def test_invalid_K(self):
        with pytest.raises(ValueError):
            OUProcess(K=1)

    def test_custom_mu_wrong_shape(self):
        with pytest.raises(ValueError):
            OUProcess(K=3, mu=np.array([0.5, 0.5]))

    def test_custom_mu_not_normalized(self):
        with pytest.raises(ValueError):
            OUProcess(K=3, mu=np.array([0.5, 0.5, 0.5]))
