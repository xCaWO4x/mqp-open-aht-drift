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
│   ├── drift_schedule.py       # (planned) controls how drift is applied across episodes
│   └── belief_tracker.py       # (planned) Bayesian filter adapter for drift-aware GPL
├── eval/                        # evaluation utilities
│   ├── metrics.py              # (planned) IQM, recovery speed, degradation curve utilities
│   ├── stability_region.py     # (planned) tools for sweeping (sigma, theta) grid
│   └── logger.py               # wandb/tensorboard logging wrapper ✓
├── experiments/
│   ├── pilot_degradation.py    # pilot sweep with RandomAgent (to be swapped for GPL)
│   ├── train_gpl.py            # GPL training on LBF (Algorithm 5) ✓
│   └── eval_drift.py           # drift evaluation: single-point + grid sweep ✓
├── configs/
│   ├── gpl_lbf.yaml            # hyperparameters for GPL on LBF
│   ├── gpl_wolfpack.yaml       # hyperparameters for GPL on Wolfpack
│   └── drift_sweep.yaml        # (sigma, theta) grid for pilot experiment
├── docs/
│   └── research_proposal.md    # formalized research proposal with citations
├── tests/
│   ├── test_ou_process.py      # unit tests for OUProcess ✓ (11 tests)
│   ├── test_drift_wrapper.py   # unit tests for DriftWrapper ✓ (6 tests)
│   ├── test_gpl_forward.py     # GPL forward pass + training tests ✓ (36 tests)
│   └── test_preprocess.py      # generic preprocess() + hidden state management ✓ (13 tests)
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

`gym.Wrapper` that applies OU drift to an LBF environment with full level injection.

- **On `reset()`**:
  1. Advances the OU process on the type-frequency simplex.
  2. Samples agent composition (type indices) via multinomial draw.
  3. Maps types to LBF levels (type 0 -> level 1, etc.).
  4. Samples food levels from configured distribution.
  5. Injects exact levels into inner env (`min=max=desired_level`).
  6. Resets inner env.
- **On `step()`**: passes through unchanged — composition is fixed within an episode.
- **Food modes**: `"fixed"` (primary: levels from {2: 0.6, 3: 0.4}) or `"coupled"` (ablation: centered on mean agent level).
- **Properties**: `.composition`, `.agent_levels`, `.food_levels`, `.ou_state`.
- **`episode_summary()`**: returns logging dict with target distribution, realized composition, agent/food levels, mean/total stats.

The inner `ForagingEnv` must be created with `min_player_level=1, max_player_level=K`
so the observation space accommodates all possible levels.

```python
from lbforaging.foraging.environment import ForagingEnv
from drift.ou_process import OUProcess
from envs.drift_wrapper import DriftWrapper

inner = ForagingEnv(players=3, min_player_level=[1,1,1], max_player_level=[3,3,3],
                    field_size=(8,8), max_num_food=3, min_food_level=[1,1,1],
                    max_food_level=[3,3,3], sight=8, max_episode_steps=200,
                    force_coop=True)
ou = OUProcess(K=3, theta=0.15, sigma=0.2, seed=42)
env = DriftWrapper(inner, ou, n_agents=3, n_food=3, food_mode="fixed", seed=42)

obs = env.reset()
print(env.agent_levels)    # e.g. [3, 1, 2]
print(env.food_levels)     # e.g. [2, 3, 2]
print(env.episode_summary())
```

### `agents/baselines/random_agent.py` — `RandomAgent`

Minimal baseline that uniformly samples from the action space, ignoring
observations. Used as the pilot experiment placeholder until a trained GPL
checkpoint is wired in (Step 15).

### `agents/gpl/` — GPL (fully implemented)

Complete implementation of Graph-based Policy Learning aligned with
Rahman et al. 2023 ([arXiv:2210.05448](https://arxiv.org/abs/2210.05448)),
Appendix A (Algorithms 2-5). PREPROCESS (C.1) is in `envs/env_utils.py`.

| Class | Paper section | Description |
|-------|---------------|-------------|
| `TypeInferenceModel` | §4.2, Eq. 7 | LSTMCell mapping B_t → type vectors θ (hidden state projected to type_dim; identity when type_dim = hidden_dim) |
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

### `experiments/eval_drift.py` — Drift evaluation

Evaluates a trained GPL checkpoint under OU-process composition drift.
Two modes of operation:

- **Single point**: Evaluate at one (sigma, theta) pair, produces per-episode CSV
  and 3-panel trajectory plot (returns, agent levels, OU type frequencies).
- **Grid sweep**: Full (sigma, theta) grid across multiple seeds, produces
  aggregate + per-episode CSVs and side-by-side mean/IQM heatmaps.

```bash
# Single point (default: sigma=0.1, theta=0.15)
python experiments/eval_drift.py --checkpoint results/gpl_lbf_train_01_paper_full/checkpoints/gpl_final.pt

# Custom sigma/theta
python experiments/eval_drift.py --checkpoint path.pt --sigma 0.2 --theta 0.3

# Full grid sweep (uses configs/drift_sweep.yaml)
python experiments/eval_drift.py --checkpoint path.pt --sweep

# Smoke test (3 episodes, reduced grid)
python experiments/eval_drift.py --checkpoint path.pt --smoke-test
```

### `configs/drift_sweep.yaml`

Default sweep configuration:

| Parameter | Value |
|-----------|-------|
| Environment | LBF 8x8, 3 agents, 3 food, force_coop=True |
| Agent types (K) | 3 (levels 1, 2, 3) |
| Food mode | "fixed" — levels from {2: 0.6, 3: 0.4} |
| Sigmas | 0.01, 0.05, 0.1, 0.2, 0.5 |
| Thetas | 0.05, 0.15, 0.3, 0.5, 1.0 |
| Episodes per grid point | 50 |
| Seeds | 5 (bootstrap CIs) |
| Max steps per episode | 200 |
| Stability threshold | 10% degradation |
| Primary metric | IQM return |

---

## Running tests

```bash
conda activate drift-aht
python -m pytest tests/ -v
```

Expected output: **83 tests passing**.

| Test file | Tests | What it checks |
|-----------|-------|----------------|
| `test_ou_process.py` | 11 | Simplex invariance, long-run mean convergence, variance scaling, input validation |
| `test_drift_wrapper.py` | 23 | Food sampling (fixed/coupled), level injection, composition drift, food modes, episode summary |
| `test_gpl_forward.py` | 36 | All GPL sub-modules: shapes, forward passes, training, persistence |
| `test_preprocess.py` | 13 | Generic `preprocess()`: B_j=[x_j;u] construction, hidden state management. Note: `preprocess_lbf`/`preprocess_wolfpack` not yet covered. |

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