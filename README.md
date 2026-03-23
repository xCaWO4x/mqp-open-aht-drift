# open-aht-drift

Robustness of open ad hoc teamwork (AHT) algorithms under stochastic population composition drift. Primary baseline: GPL (Graph-based Policy Learning, Rahman et al. ICML 2021 / JMLR 2023).

## Setup

### One-time setup

These steps create the environment, install all dependencies, and patch compatibility issues. Run once on a new machine.

```bash
# 1. Create conda environment
conda create -n drift-aht python=3.8 -y

# 2. Activate it
conda activate drift-aht

# 3. Pin build tools (newer versions break gym 0.21.0)
pip install "setuptools==65.5.0" "wheel==0.38.4"

# 4. Core packages
pip install numpy scipy matplotlib tensorboard wandb

# 5. PyTorch (CPU / Apple Silicon MPS)
pip install torch torchvision torchaudio

# 6. Gym (pinned — later versions break MPE/LBF interface)
pip install "gym==0.21.0"

# 7. Level-Based Foraging
pip install lbforaging

# 8. Clone GPL repo (contains Wolfpack environment)
git clone https://github.com/uoe-agents/GPL.git ../GPL

# 9. Install Wolfpack environment
#    First, edit ../GPL/Open_Experiments/Wolfpack/env/setup.py:
#    change  install_requires=['gym', 'torch', 'numpy', 'pygame','dgl==0.3']
#    to      install_requires=['gym', 'torch', 'numpy', 'pygame']
pip install -e ../GPL/Open_Experiments/Wolfpack/env

# 10. Install compatible DGL (original pin dgl==0.3 is unavailable)
pip install "dgl==0.9.1"
```

#### DGL API patch (required once)

In `../GPL/Open_Experiments/Wolfpack/env/Wolfpack_gym/envs/wolfpack_penalty_single_assets/QNetwork.py`, replace all occurrences of:
- `graph.batch_num_edges` → `graph.batch_num_edges() if callable(graph.batch_num_edges) else graph.batch_num_edges`
- `graph.batch_num_nodes` → `graph.batch_num_nodes() if callable(graph.batch_num_nodes) else graph.batch_num_nodes`

This fixes the DGL 0.3 → 0.9 API change where these became methods instead of properties.

### Per-session setup

Run at the start of each terminal session.

```bash
conda activate drift-aht
export DGLBACKEND=pytorch
```

### Verify installation

```bash
# LBF
python -c "
import gymnasium
env = gymnasium.make('Foraging-8x8-2p-1f-v3')
obs, info = env.reset()
for _ in range(5):
    actions = [env.action_space[i].sample() for i in range(len(env.action_space))]
    env.step(actions)
env.close()
print('LBF OK')
"

# Wolfpack
python -c "
import gym, Wolfpack_gym
env = gym.make('Adhoc-wolfpack-v5')
obs = env.reset()
for _ in range(5):
    env.step(env.action_space.sample())
env.close()
print('Wolfpack OK')
"
```

---

## Repository structure

```
open-aht-drift/
├── envs/
│   ├── drift_wrapper.py        # OU drift process wrapper — core contribution
│   └── env_utils.py            # PREPROCESS (C.1) + LBF/Wolfpack obs adapters ✓
├── agents/
│   ├── gpl/                    # GPL model code (fully implemented)
│   │   ├── type_inference.py   # LSTM-based type embedding (§4.2, Eq. 7)
│   │   ├── agent_model.py      # RFM-based teammate action prediction (§4.4, Eqs. 11-13)
│   │   ├── joint_action_value.py  # Coordination graph Q-value (§4.3, Eqs. 8-10)
│   │   └── gpl_agent.py        # top-level GPL agent (Algorithms 2-5)
│   └── baselines/
│       └── random_agent.py     # sanity check baseline
├── drift/
│   ├── ou_process.py           # OU process over simplex ✓
│   ├── drift_schedule.py       # controls how drift is applied across episodes
│   └── belief_tracker.py       # placeholder for future Bayesian filter adapter
├── eval/
│   ├── metrics.py              # IQM, recovery speed, degradation curve utilities
│   ├── stability_region.py     # tools for sweeping (sigma, theta) grid
│   └── logger.py               # wandb/tensorboard logging wrapper
├── experiments/
│   ├── pilot_degradation.py    # FIRST SCRIPT TO RUN: GPL under OU drift sweep
│   ├── train_gpl.py            # GPL training on standard LBF/Wolfpack
│   └── eval_drift.py           # evaluation under drift given a trained checkpoint
├── configs/
│   ├── gpl_lbf.yaml            # hyperparameters for GPL on LBF
│   ├── gpl_wolfpack.yaml       # hyperparameters for GPL on Wolfpack
│   └── drift_sweep.yaml        # (sigma, theta) grid for pilot experiment
├── tests/
│   ├── test_ou_process.py      # unit tests for OUProcess ✓ (11 tests)
│   ├── test_drift_wrapper.py   # unit tests for DriftWrapper ✓ (6 tests)
│   ├── test_gpl_forward.py     # GPL forward pass + training tests ✓ (37 tests)
│   └── test_preprocess.py      # PREPROCESS + hidden state management ✓ (12 tests)
├── requirements.txt
└── setup.py
```

---

## Implemented modules

### `drift/ou_process.py` — `OUProcess`

Ornstein-Uhlenbeck process over the K-simplex. Models smoothly drifting
agent-type frequencies across episodes.

**Algorithm**: Euler-Maruyama discretisation in the ambient space, followed
by Euclidean projection onto the simplex (Duchi et al. 2008):

```
x ← x + θ(μ − x) dt + σ√dt · N(0, I)
x ← project_onto_simplex(x)
```

| Parameter | Description |
|-----------|-------------|
| `K` | Number of agent types |
| `theta` | Mean-reversion rate — higher pulls state back to `mu` faster |
| `sigma` | Noise scale — higher gives larger random perturbations |
| `mu` | Target mean on the simplex (default: uniform) |
| `dt` | Euler-Maruyama timestep (default: 0.01) |

**Key methods**:
- `step()` — advance one timestep, return new state
- `reset()` — return to `mu` with small noise
- `sample_composition(n_agents)` — draw agent types i.i.d. from current state

### `envs/drift_wrapper.py` — `DriftWrapper`

`gym.Wrapper` that applies OU drift to any LBF or Wolfpack environment.

- **On `reset()`**: advances the OU process, samples a new team composition,
  resets the inner environment.
- **On `step()`**: passes through unchanged — composition is fixed within an episode.
- **Properties**: `.composition` (list of type indices), `.ou_state` (frequency vector).

```python
from drift.ou_process import OUProcess
from envs.drift_wrapper import DriftWrapper

ou = OUProcess(K=3, theta=0.15, sigma=0.2, seed=42)
env = DriftWrapper(inner_env, ou_process=ou, n_agents=4)

obs = env.reset()          # OU advances, new composition sampled
print(env.composition)     # e.g. [0, 2, 0, 1]
print(env.ou_state)        # e.g. [0.45, 0.30, 0.25]
```

### `agents/baselines/random_agent.py` — `RandomAgent`

Minimal baseline that uniformly samples from the action space, ignoring
observations. Used as a sanity-check placeholder in the pilot experiment
before GPL is implemented.

### `agents/gpl/` — GPL (fully implemented)

Complete implementation of Graph-based Policy Learning aligned with
Rahman et al. 2023 ([arXiv:2210.05448](https://arxiv.org/abs/2210.05448)),
Appendix A (Algorithms 2-5) and C.1 (PREPROCESS).

| Class | Paper section | Description |
|-------|---------------|-------------|
| `TypeInferenceModel` | §4.2, Eq. 7 | LSTMCell mapping B_t → type vectors θ (hidden state IS the type) |
| `AgentModel` | §4.4, Eqs. 11-13 | RFM_ζ(θ', c') + MLP_η → teammate action distributions |
| `JointActionValueModel` | §4.3, Eqs. 8-10 | MLP_β (individual) + MLP_δ (low-rank pairwise) Q-values |
| `GPLAgent` | §4.1-4.6, Alg. 2-5 | Top-level agent: `act()`, `compute_qv/qjoint/pteam()`, `train_step_online()`, `update()`, `save()`, `load()` |

### `envs/env_utils.py` — PREPROCESS (Appendix C.1)

Converts raw environment state into GPL input format: B_j = [x_j ; u].
Handles open agent sets with LSTM hidden state carry/zero-init/removal.

### `experiments/pilot_degradation.py` — Pilot degradation sweep

First experiment script. Sweeps a grid of (sigma, theta) OU parameters and
measures mean episodic return under each drift regime.

- Loads config from `configs/drift_sweep.yaml`
- Wraps LBF environment with `DriftWrapper`
- Runs N episodes per grid point with a `RandomAgent` (placeholder for GPL)
- Saves results to `results/pilot/degradation_grid.csv`
- Generates a heatmap at `results/pilot/degradation_heatmap.png`
- Optionally logs to wandb

```bash
# Full sweep (50 episodes per grid point, wandb logging)
python experiments/pilot_degradation.py

# Quick sanity check (2 episodes, no wandb)
python experiments/pilot_degradation.py --dry-run

# Custom config
python experiments/pilot_degradation.py --config configs/my_sweep.yaml
```

### `configs/drift_sweep.yaml`

Default sweep configuration:

| Parameter | Value |
|-----------|-------|
| Environment | `Foraging-8x8-2p-1f-v3` (LBF, 2 players, 1 food) |
| Agent types (K) | 3 |
| Sigmas | 0.01, 0.05, 0.1, 0.2, 0.5 |
| Thetas | 0.05, 0.15, 0.3, 0.5, 1.0 |
| Episodes per grid point | 50 |
| Max steps per episode | 200 |

---

## Running tests

```bash
conda activate drift-aht
python -m pytest tests/ -v
```

Expected output: **66 tests passing**.

| Test file | Tests | What it checks |
|-----------|-------|----------------|
| `test_ou_process.py` | 11 | Simplex invariance, long-run mean convergence, variance scaling, input validation |
| `test_drift_wrapper.py` | 6 | Composition changes across episodes, stable within episode, gym interface |
| `test_gpl_forward.py` | 37 | All GPL sub-modules: shapes, forward passes, training, persistence |
| `test_preprocess.py` | 12 | B_j=[x_j;u] construction, hidden state management for open agent sets |

---

## Verification checklist

Run these after setup or any major change:

```bash
# 1. All unit tests
python -m pytest tests/ -v

# 2. LBF import
python -c "import lbforaging; print('LBF OK')"

# 3. OU process sanity check
python -c "from drift.ou_process import OUProcess; o = OUProcess(theta=0.5, sigma=0.1, K=3); o.step(); print(o.x)"

# 4. Pilot dry-run (no crash, produces CSV + heatmap)
python experiments/pilot_degradation.py --dry-run
```