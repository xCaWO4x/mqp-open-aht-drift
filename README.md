# open-aht-drift

Robustness of open ad hoc teamwork (AHT) algorithms under stochastic population composition drift. Primary baseline: GPL (Graph-based Policy Learning, Rahman et al. ICML 2021 / JMLR 2023).

---

## Setup

```bash
# Create and activate environment
conda create -n drift-aht python=3.8 -y
conda activate drift-aht

# Pin build tools (newer versions break gym 0.21.0)
pip install "setuptools==65.5.0" "wheel==0.38.4"

# Core packages
pip install numpy scipy matplotlib tensorboard wandb

# PyTorch
pip install torch torchvision torchaudio

# Gym (pinned)
pip install "gym==0.21.0"

# Level-Based Foraging
pip install lbforaging
```

Per session:
```bash
conda activate drift-aht
```

---

## Repository structure

```
open-aht-drift/
├── agents/
│   ├── gpl/                    # GPL implementation (Rahman et al. 2023)
│   │   ├── type_inference.py   # LSTM type embedding
│   │   ├── agent_model.py      # RFM teammate action prediction
│   │   ├── joint_action_value.py  # Coordination graph Q-values
│   │   └── gpl_agent.py        # Top-level agent (Algorithms 2–5)
│   └── baselines/
│       └── random_agent.py
├── drift/
│   └── ou_process.py           # OU process over K-simplex
├── envs/
│   ├── drift_wrapper.py        # Gym wrapper: OU drift + level injection
│   └── env_utils.py            # PREPROCESS (Appendix C.1), LBF obs parsing
├── eval/
│   └── logger.py               # TensorBoard / wandb logging
├── experiments/
│   ├── train_gpl.py            # GPL training loop (Algorithm 5)
│   ├── eval_drift.py           # Drift evaluation sweep
│   └── analyze_capability_confound.py  # Capability vs drift decomposition
├── configs/
│   ├── gpl_lbf.yaml            # Q1/Q2 baseline config (Rahman paper)
│   ├── gpl_lbf_q3_hardened.yaml   # Q3_hardened / Q4_hardened (partial obs, 4p, force coop)
│   ├── drift_sweep.yaml        # (σ, θ) grid for Q2/Q4
│   └── gpl_wolfpack.yaml       # Future: Wolfpack
├── scripts/slurm/
│   ├── q1_train.slurm          # Q1 baseline training
│   ├── q2_drift_eval.slurm     # Q2 baseline + drift
│   ├── q3_hardened_train.slurm # Q3_hardened training
│   ├── q4_drift_eval.slurm     # Q4_hardened + drift
│   ├── submit_training.sh      # Submit Q1 + Q3_hardened
│   └── submit_drift_eval.sh    # Submit Q2 + Q4
├── tests/
│   ├── test_ou_process.py      # 11 tests
│   ├── test_drift_wrapper.py   # 23 tests
│   ├── test_gpl_forward.py     # 36 tests
│   └── test_preprocess.py      # 19 tests
├── docs/
│   └── experiment_artifacts.md # Result layout and artifact map
├── INSTRUCTIONS.md             # Project context and next steps
├── requirements.txt
└── setup.py
```

---

## Experiment layout

|  | Stationary | Drift |
|--|-----------|-------|
| **Baseline** (Rahman config) | Q1 | Q2 |
| **Hardened** (partial obs, latent types, 4p, force coop) | Q3 | Q4 |

```bash
# Train Q1 + Q3
bash scripts/slurm/submit_training.sh

# After training: eval Q2 + Q4
bash scripts/slurm/submit_drift_eval.sh
```

See `docs/experiment_artifacts.md` for full artifact map and result directory layout.

---

## Running tests

```bash
conda activate drift-aht
python -m pytest tests/ -v
```

Expected: **89 tests passing**.

| File | Tests | Coverage |
|------|-------|----------|
| `test_ou_process.py` | 11 | Simplex invariance, mean convergence, variance scaling |
| `test_drift_wrapper.py` | 23 | Food sampling, level injection, composition drift |
| `test_gpl_forward.py` | 36 | GPL sub-modules: shapes, forward passes, training, persistence |
| `test_preprocess.py` | 19 | PREPROCESS, LBF obs parsing, hidden state management |
