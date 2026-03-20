"""
Smoke test for GPL forward pass.

Since the GPL modules are currently stubs, this test verifies that:
  1. The classes can be imported without error.
  2. Instantiation raises NotImplementedError (confirming stubs are in place).
  3. Once implemented, the test structure is ready for real forward-pass checks.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agents.gpl.type_inference import TypeInferenceModel
from agents.gpl.agent_model import AgentModel
from agents.gpl.joint_action_value import JointActionValueModel
from agents.gpl.gpl_agent import GPLAgent


class TestGPLStubs:
    """Verify all GPL stubs import and raise NotImplementedError."""

    def test_type_inference_model_is_stub(self):
        with pytest.raises(NotImplementedError):
            TypeInferenceModel(obs_dim=10, action_dim=5)

    def test_agent_model_is_stub(self):
        with pytest.raises(NotImplementedError):
            AgentModel(type_dim=32)

    def test_joint_action_value_model_is_stub(self):
        with pytest.raises(NotImplementedError):
            JointActionValueModel(obs_dim=10)

    def test_gpl_agent_is_stub(self):
        with pytest.raises(NotImplementedError):
            GPLAgent(obs_dim=10, action_dim=5)
