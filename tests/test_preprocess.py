"""
Unit tests for PREPROCESS (Appendix C.1) and environment utilities.

Covers:
  - B_j = [x_j; u] construction
  - LSTM hidden state management for open agent sets
  - Edge cases: single agent, all agents replaced, agent reordering
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from envs.env_utils import preprocess


# ======================================================================
# Fixtures
# ======================================================================

HIDDEN_DIM = 64


@pytest.fixture
def simple_obs():
    """Raw obs with 3 agents (4 features each) + 3 shared features = 15 total."""
    return np.arange(15, dtype=np.float32)


@pytest.fixture
def agent_slices_3():
    """Feature slices for 3 agents, 4 features each."""
    return {0: slice(0, 4), 1: slice(4, 8), 2: slice(8, 12)}


@pytest.fixture
def shared_slice():
    return slice(12, 15)


# ======================================================================
# B_j = [x_j ; u] construction
# ======================================================================

class TestPreprocessConstruction:

    def test_output_shape(self, simple_obs, agent_slices_3, shared_slice):
        B, hidden, agent_ids = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        assert B.shape == (3, 7)  # 4 agent features + 3 shared
        assert hidden[0].shape == (3, HIDDEN_DIM)
        assert hidden[1].shape == (3, HIDDEN_DIM)
        assert agent_ids == [0, 1, 2]

    def test_bj_is_xj_concat_u(self, simple_obs, agent_slices_3, shared_slice):
        """B_j should be [x_j ; u] for each agent."""
        B, _, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        u = simple_obs[12:15]
        for j in range(3):
            x_j = simple_obs[j * 4:(j + 1) * 4]
            expected = np.concatenate([x_j, u])
            np.testing.assert_array_almost_equal(B[j].numpy(), expected)

    def test_shared_features_same_for_all(self, simple_obs, agent_slices_3, shared_slice):
        """The shared portion of B should be identical across all agents."""
        B, _, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        # Last 3 elements of each row should be the shared features
        for j in range(3):
            np.testing.assert_array_equal(
                B[j, 4:].numpy(), B[0, 4:].numpy()
            )

    def test_agent_features_differ(self, simple_obs, agent_slices_3, shared_slice):
        """Per-agent portions of B should differ (since obs values differ)."""
        B, _, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        assert not torch.equal(B[0, :4], B[1, :4])
        assert not torch.equal(B[1, :4], B[2, :4])

    def test_single_agent(self):
        """Should work with a single agent."""
        obs = np.array([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
        slices = {0: slice(0, 3)}
        B, hidden, ids = preprocess(
            obs, slices, slice(3, 5), hidden_dim=HIDDEN_DIM,
        )
        assert B.shape == (1, 5)
        assert ids == [0]
        expected = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(B[0].numpy(), expected)

    def test_curr_agent_ids_ordering(self, simple_obs, agent_slices_3, shared_slice):
        """If curr_agent_ids is given, B rows should follow that order."""
        B, _, ids = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            curr_agent_ids=[2, 0, 1],
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [2, 0, 1]
        # First row should be agent 2's features
        x_2 = simple_obs[8:12]
        u = simple_obs[12:15]
        expected = np.concatenate([x_2, u])
        np.testing.assert_array_almost_equal(B[0].numpy(), expected)


# ======================================================================
# Hidden state management for openness
# ======================================================================

class TestPreprocessHiddenStates:

    def test_no_prior_state_zeros(self, simple_obs, agent_slices_3, shared_slice):
        """Without prior hidden state, should return zeros."""
        _, hidden, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            hidden_dim=HIDDEN_DIM,
        )
        assert (hidden[0] == 0).all()
        assert (hidden[1] == 0).all()

    def test_carry_forward_persistent_agents(self, simple_obs, agent_slices_3, shared_slice):
        """Hidden states should carry forward for agents present in both steps."""
        prev_h = torch.randn(3, HIDDEN_DIM)
        prev_c = torch.randn(3, HIDDEN_DIM)
        prev_ids = [0, 1, 2]

        _, hidden, _ = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 1, 2],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert torch.equal(hidden[0], prev_h)
        assert torch.equal(hidden[1], prev_c)

    def test_new_agent_gets_zeros(self, simple_obs, agent_slices_3, shared_slice):
        """A newly appearing agent should get zero-initialised hidden state."""
        prev_h = torch.randn(2, HIDDEN_DIM)
        prev_c = torch.randn(2, HIDDEN_DIM)
        prev_ids = [0, 1]

        _, hidden, ids = preprocess(
            simple_obs, agent_slices_3, shared_slice,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 1, 2],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        # Agents 0 and 1 carried forward
        assert torch.equal(hidden[0][0], prev_h[0])
        assert torch.equal(hidden[0][1], prev_h[1])
        # Agent 2 is new → zeros
        assert (hidden[0][2] == 0).all()
        assert (hidden[1][2] == 0).all()

    def test_departed_agent_removed(self):
        """Departed agent's hidden state should not appear in output."""
        obs = np.arange(11, dtype=np.float32)  # 2 agents × 4 + 3 shared
        slices = {0: slice(0, 4), 1: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(3, HIDDEN_DIM)
        prev_c = torch.randn(3, HIDDEN_DIM)
        prev_ids = [0, 1, 2]  # agent 2 was here before

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 1],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [0, 1]
        assert hidden[0].shape == (2, HIDDEN_DIM)
        # Agent 0 and 1 carried forward from their old positions
        assert torch.equal(hidden[0][0], prev_h[0])
        assert torch.equal(hidden[0][1], prev_h[1])

    def test_all_agents_replaced(self):
        """If all agents change, all hidden states should be zero."""
        obs = np.arange(11, dtype=np.float32)
        slices = {3: slice(0, 4), 4: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(2, HIDDEN_DIM)
        prev_c = torch.randn(2, HIDDEN_DIM)
        prev_ids = [0, 1]  # completely different

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[3, 4],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [3, 4]
        assert (hidden[0] == 0).all()
        assert (hidden[1] == 0).all()

    def test_agent_reordering(self):
        """Hidden states should follow agent IDs, not positions."""
        obs = np.arange(11, dtype=np.float32)
        slices = {1: slice(0, 4), 0: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(2, HIDDEN_DIM)
        prev_c = torch.randn(2, HIDDEN_DIM)
        prev_ids = [0, 1]  # agent 0 was at index 0, agent 1 at index 1

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[1, 0],  # swapped order
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [1, 0]
        # Agent 1 (now at position 0) should have agent 1's old hidden state
        assert torch.equal(hidden[0][0], prev_h[1])
        # Agent 0 (now at position 1) should have agent 0's old hidden state
        assert torch.equal(hidden[0][1], prev_h[0])

    def test_mixed_new_and_persistent(self):
        """Some agents persist, some depart, some are new."""
        obs = np.arange(11, dtype=np.float32)
        slices = {0: slice(0, 4), 5: slice(4, 8)}
        shared = slice(8, 11)

        prev_h = torch.randn(3, HIDDEN_DIM)
        prev_c = torch.randn(3, HIDDEN_DIM)
        prev_ids = [0, 1, 2]
        # Agent 0 persists, agents 1 & 2 depart, agent 5 is new

        _, hidden, ids = preprocess(
            obs, slices, shared,
            prev_agent_ids=prev_ids,
            curr_agent_ids=[0, 5],
            prev_hidden=(prev_h, prev_c),
            hidden_dim=HIDDEN_DIM,
        )
        assert ids == [0, 5]
        # Agent 0: carried forward
        assert torch.equal(hidden[0][0], prev_h[0])
        # Agent 5: new → zeros
        assert (hidden[0][1] == 0).all()
        assert (hidden[1][1] == 0).all()
