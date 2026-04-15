# GPL Robustness Under Population Drift in Open Ad Hoc Teamwork

Research document for the MQP. Covers motivation, background, methodology, experimental design, and proposed remedy.

---

## 1. Introduction and Motivation

**Open ad hoc teamwork (AHT)** requires an agent to cooperate with previously unseen teammates whose behaviors, capabilities, and even number may change across episodes. The standard AHT protocol (Rahman et al., 2021; 2023) trains a learner agent (agent 0) using a framework like GPL, then evaluates it with random teammates whose types are drawn i.i.d. from a fixed (stationary) distribution.

Real-world teams are not stationary. Personnel rotate, skill distributions shift, and the population the agent encounters tomorrow may differ systematically from what it trained on. We model this **non-stationary population composition** as a continuous stochastic process — specifically, an Ornstein-Uhlenbeck (OU) process over the type-frequency simplex — and ask:

> **How robust is GPL to smooth, persistent drift in the team composition it encounters at deployment time?**

This is distinct from the open-set generalization studied in prior AHT work (which asks "can the agent handle a teammate *type* it has never seen?"). Here every type was present during training; what changes is the *mixture* over types across episodes. The drift is temporally correlated — not random episode-to-episode variation, but a trend that persists over tens or hundreds of episodes.

---

## 2. Background

### 2.1 GPL (Graph-based Policy Learning)

GPL (Rahman et al., ICML 2021 / JMLR 2023, arXiv:2006.10412 / 2210.05448) is a three-component architecture for open ad hoc teamwork:

1. **TypeInferenceModel** (Section 4.2, Eq. 7): An LSTM that maps per-agent preprocessed state features B_t to continuous type vectors theta. The LSTM hidden state *is* the type vector — types are continuous, not discrete labels, enabling generalization to novel teammates. Two separate copies are maintained: alpha_Q (feeding the Q-network) and alpha_q (feeding the agent model), to prevent loss interference.

2. **AgentModel** (Section 4.4, Algorithm 4 — PTEAM): A Graph Neural Network (RFM) followed by an MLP. Takes type vectors theta and LSTM cell states c as node features, performs message passing, and outputs per-agent action probability distributions q(a^j | s). Trained via negative log-likelihood of observed teammate actions (Eq. 15).

3. **JointActionValueModel** (Section 4.3, Eq. 8-10): A coordination-graph-based Q-network that factorizes the joint action value into singular terms (MLP_beta) and low-rank pairwise terms (MLP_delta). Both take concatenated (theta^j, theta^i) as input. The ego agent's action is selected by marginalizing over teammate actions weighted by the agent model's predictions (Eq. 14).

**Training** follows Algorithm 5: synchronous online learning with 16 parallel environments, gradient accumulation over t_update=4 steps, and Polyak soft target updates (tau=1e-3) every step. The learner (agent 0) uses GPL; all teammates act randomly (standard open AHT evaluation protocol).

**PREPROCESS** (Appendix C.1): Splits raw environment observations into per-agent features x_j and shared features u, producing B_j = [x_j; u] for each agent j. This ensures each agent's type vector depends only on its own trajectory plus global context.

### 2.2 Level-Based Foraging (LBF)

LBF is a grid-world cooperative foraging task. Each agent and each food item has an integer level. Food can only be collected when agents adjacent to it have total levels >= the food level. The environment returns per-agent ego-centric observations containing food positions/levels and agent positions/levels.

We use K=3 agent types mapping directly to LBF levels {1, 2, 3}. Food is sampled from a fixed distribution: level 2 (60%) and level 3 (40%), independent of agent composition. This creates a capability-task mismatch when the population drifts toward low-level agents.

### 2.3 Ornstein-Uhlenbeck Drift Model

We model population composition drift as an OU process on the K-simplex:

```
x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * N(0, I)
x_{t+1} = project_simplex(x_{t+1})
```

- **x_t** is the K-dimensional type-frequency vector on the probability simplex
- **theta** is the mean-reversion rate (higher = faster return to mu)
- **sigma** is the noise scale (higher = more volatile drift)
- **mu** is the target mean (uniform 1/K by default)
- **dt** = 0.01 is the discrete timestep
- Projection onto the simplex uses the Euclidean algorithm of Duchi et al. (2008)

At each episode reset, the OU process advances one step, producing a new type-frequency vector. Agent types are then sampled i.i.d. from this distribution. Within an episode, the composition is fixed.

The OU model captures key properties of real population drift: temporal correlation (today's composition is similar to yesterday's), mean-reversion (extreme imbalances are transient), and smooth variation (no discontinuous jumps). The two parameters (sigma, theta) span a meaningful space from near-stationary (low sigma) to highly volatile (high sigma, low theta).

---

## 3. Experimental Design: Four Quadrants + Inference Variant

### 3.1 Quadrant Layout

|  | Stationary (no drift) | Drift |
|---|---|---|
| **Baseline** (Rahman paper config) | **Q1** | **Q2** |
| **Hardened** (partial obs, latent types, 4p, force coop) | **Q3** | **Q4** |
| **Hardened + inference** (aux head + EMA tracker) | **Q3-inf** | **Q4-inf** |

- Q1 and Q3 (and Q3-inf) are *training* runs under stationary composition.
- Q2, Q4, and Q4-inf are *evaluation-only* runs using the corresponding trained checkpoints under OU drift across a sweep of (sigma, theta) values.
- The drift sweep grid: 10 sigma values x 5 theta values x 100 episodes x 5 seeds = 25,000 episodes per quadrant.

### 3.2 Drift Sweep Grid

```
sigmas: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
thetas: [0.05, 0.15, 0.3, 0.5, 1.0]
```

sigma=0 row serves as the stationary baseline within each sweep. Degradation is measured as fractional IQM loss relative to this baseline: `degradation = 1 - IQM(sigma, theta) / IQM_baseline`.

A grid point is "stable" if degradation < 10%. The fraction of stable grid points characterizes the policy's robustness envelope.

### 3.3 Q1 — Baseline (Stationary Training)

Matches the Rahman et al. GPL source configuration for LBF:

| Setting | Value |
|---------|-------|
| Grid | 8x8 |
| Agents | 3 |
| Sight | 8 (full grid) |
| observe_agent_levels | true |
| force_coop | false |
| obs_dim | 12 (3 agent features + 9 food features) |
| Episodes | 128,000 |
| Parallel envs | 16 |

Config: `configs/gpl_lbf.yaml`

### 3.4 Q2 — Baseline + Drift

Loads Q1's trained checkpoint and evaluates under the full drift sweep grid. No retraining.

### 3.5 Q3 — Hardened (Stationary Training)

The baseline proves too forgiving: GPL with full observability and optional cooperation never needs to infer latent teammate types, so composition drift barely matters. Q3 applies four "nerfs" to the information structure to make teammate modeling both necessary and fragile:

| Setting | Q1 (baseline) | Q3_hardened | Rationale |
|---------|:---:|:---:|-----------|
| sight | 8 (full grid) | 3 (partial) | Forces reliance on type inference over direct observation |
| observe_agent_levels | true | false | Makes teammate type genuinely latent — must infer from behavior |
| n_agents | 3 | 4 | More composition complexity: C(6,2)=15 vs C(4,2)=10 distinct compositions; coordination graph has 6 pairwise terms vs 3 |
| force_coop | false | true | No agent can solo-collect any food — cooperation is mandatory |

With observe_agent_levels=false, LBF drops the level feature entirely from agent observations, reducing per-agent features from 3 (y, x, level) to 2 (y, x), and obs_dim from 12 to 11.

The model architecture (hidden_dim, type_dim, n_gnn_layers, etc.) and training hyperparameters remain identical to Q1. We test the *same algorithm* under harder conditions, not a different algorithm.

Config: `configs/gpl_lbf_q3_hardened.yaml`

### 3.6 Q4 — Hardened + Drift

Loads Q3's trained checkpoint and evaluates under the same drift sweep grid as Q2.

The thesis prediction: Q4 will show significantly more degradation than Q2, because the hardened information structure means GPL's type inference is both more critical and more fragile under compositional shift.

### 3.7 Capability Confound Analysis

A key methodological concern: observed performance changes under drift may reflect shifts in *effective team strength* rather than the agent's inability to handle non-stationarity. When sigma is high, the OU process may produce compositions skewed toward low-level agents, and performance drops simply because the team is weaker — not because GPL fails to adapt.

We address this via `experiments/analyze_capability_confound.py`, which runs:

1. **OLS regressions** with multiple specifications: return ~ mean_level + sigma + theta + sigma:theta, return ~ composition_categorical + sigma + theta, etc. If sigma/theta coefficients become non-significant after controlling for mean_level, the "drift effect" is a capability confound.

2. **Same-mean-level comparisons**: For episodes with matched mean agent levels, compare returns across different (sigma, theta) values. True non-stationarity robustness failures would show up here.

3. **Variance decomposition**: What fraction of return variance is explained by composition vs. drift parameters?

4. **Success rate analysis**: P(return > 0) conditioned on composition and drift parameters.

Additionally, a "coupled food" ablation (`--food-mode coupled`) scales food difficulty with agent levels, isolating the non-stationarity effect from task difficulty changes. In coupled mode, food levels track the team composition, so a weaker team faces easier food — removing the capability confound.

---

## 4. Proposed Remedy: Auxiliary Inference + EMA Belief Tracking

### 4.1 Problem Diagnosis

Under the hardened settings, GPL's type inference LSTM must infer teammate types purely from behavioral trajectories, without the level feature as a shortcut. When the population composition drifts, the distribution over type vectors that the LSTM encounters at deployment differs from training. The LSTM was never trained to track *changes* in the population — it sees each episode independently and has no cross-episode memory.

Two specific failure modes:

1. **Type embedding degradation**: With observe_agent_levels=false, the type embeddings may not encode meaningful teammate structure at all — there is no direct supervisory signal forcing them to represent level. The agent model learns to predict teammate actions, but the type vectors may encode action-predictive features (e.g., position patterns) rather than the underlying capability.

2. **No population context**: GPL has no mechanism to condition its policy on *which region of composition space it is currently in*. Even if the LSTM perfectly identifies individual teammate types within an episode, the policy cannot use cross-episode trends to anticipate what kinds of teammates it will encounter next.

### 4.2 Remedy 1: Auxiliary Level Prediction Head

**Architecture**: A small MLP (type_dim -> 64 -> ReLU -> 3) attached to the agent model's type embeddings. Predicts teammate level class from the type vector.

**Training**: Cross-entropy loss against privileged ground-truth LBF levels. This follows the CTDE (Centralized Training, Decentralized Execution) paradigm — privileged labels are available during training but hidden at execution time. The auxiliary loss gradient flows back through the type embeddings (alpha_q network), forcing them to encode meaningful teammate structure.

**Key details**:
- Levels are 1-indexed in LBF; converted to 0-indexed internally for cross-entropy
- Loss weight: 0.1 relative to the agent model loss
- Aux head parameters are added to the agent model optimizer via `add_param_group`
- Only applied to teammate type embeddings (not the learner's own)
- A separate forward pass through type_net_agent generates embeddings for the aux loss, avoiding interference with the main LSTM hidden state update

**Implementation**: `agents/gpl/auxiliary_head.py` — `AuxiliaryLevelHead`

### 4.3 Remedy 2: EMA Belief Tracker

**Architecture**: An exponential moving average of the mean type embedding across episodes. Produces a fixed-size context vector (dim=16) that summarizes recent population composition.

**Mechanism**: After each episode, the mean type embedding across all agents and timesteps is computed. The EMA is updated:

```
ema_{t+1} = (1 - alpha) * ema_t + alpha * mean_emb_t
```

with alpha=0.1 (slow adaptation, giving ~10 episodes of effective memory).

The EMA context vector is concatenated to every agent's observation input, expanding obs_dim from 11 (base) to 27 (11 + 16). This is done by `augment_obs()`, which tiles the context across all N agents.

**Crucially, the policy must be trained with the EMA visible** for it to learn to use the signal. This is not a post-hoc inference-time addition — the EMA is present during Q3-inf training. During the first episode the EMA is zero-initialized; it begins to carry signal from the second episode onward.

**Implementation**: `drift/ema_tracker.py` — `EMABeliefTracker`

### 4.4 Combined: GPLAgentInf

`GPLAgentInf` extends `GPLAgent` with both additions:

- `__init__`: Sets effective_obs_dim = obs_dim + ema_dim. Initializes aux_head and ema_tracker. Adds aux head params to agent model optimizer.
- `augment_obs(B_np)`: Tiles EMA context and concatenates to observation batch.
- `train_step_online_inf(...)`: Runs standard Algorithm 5 training step (via super()), then runs a separate forward pass through type_net_agent for the auxiliary loss, and accumulates type embeddings for EMA update. On episode end, updates the EMA.
- `act_inf(B_np)`: EMA-augmented action selection.
- `advance_hidden_inf(B_np)`: EMA-augmented hidden state advance for evaluation.
- `end_episode_ema()`: Finalizes EMA update at episode end during evaluation.
- `save()`/`load()`: Extended to persist aux_head weights and ema_tracker state.

**Implementation**: `agents/gpl/gpl_agent_inf.py`

### 4.5 Connection to Literature

The auxiliary inference head is inspired by several recent AHT methods that use privileged supervision:

- **ODITS** (Gu et al., ICLR 2022): Uses an "open-decoding information" module with auxiliary losses to improve type inference in open teams. Our auxiliary head is a simplified version — single head predicting level class rather than full teammate policy reconstruction.

- **Fastap** (Chen & Xu, AAMAS 2023): Teacher-student distillation where a teacher with full information supervises the student's type inference. Our CTDE auxiliary loss achieves a similar effect without a separate teacher network.

- **POAM** (2024): Population-conditioned policy that explicitly conditions on population statistics. Our EMA tracker serves an analogous role but uses a learned representation (type embedding averages) rather than hand-crafted population features.

- **SMPE²** (Wang et al., ICML 2025): Self-supervised multi-agent prediction with environment-enhanced learning. Uses environment structure to improve teammate modeling — complementary to our auxiliary supervision approach.

The EMA belief tracker is a lightweight form of **population context** that avoids the complexity of full population-conditioned policies (deferred to future work). It provides a first-order summary of recent team composition that the policy can condition on, enabling implicit drift detection without explicit change-point detection or non-stationary RL machinery.

### 4.6 Evaluation Plan for the Remedy

**Q3-inf vs Q3** (stationary): Measures whether the auxiliary inference improves GPL's performance even without drift, by forcing type embeddings to be more structured. The cost is training overhead (aux head forward/backward + EMA bookkeeping).

**Q4-inf vs Q4** (drift): The critical comparison. If the EMA tracker provides useful population context under drift, Q4-inf should show less degradation than Q4 across the (sigma, theta) grid — particularly at moderate sigma values where drift is persistent but not overwhelming.

**Q4-inf vs Q2** (cross-condition): Tests whether the hardened + inference setup under drift can match or approach the baseline's drift robustness. If so, the remedy successfully recovers the robustness that the hardened nerfs removed.

---

## 5. Implementation Details

### 5.1 Observation Format

LBF observations are per-agent ego-centric. The format is **food first, then agents**:

```
[food_0(y,x,level), food_1(y,x,level), ..., self(y,x,level), other_0(y,x,level), ...]
```

With observe_agent_levels=false, agent features become (y,x) only — 2 features per agent instead of 3. The PREPROCESS function in `envs/env_utils.py` handles both modes.

### 5.2 Hidden State Management

- `act()` selects an action but **does not update** LSTM hidden states. This is by design — `train_step_online()` is the sole updater during training (Algorithm 5 lines 14-27 all start from the same h_{t-1}).
- During **evaluation**, `advance_hidden()` (or `advance_hidden_inf()`) must be called after each `act()` to keep the LSTM context current.
- With 16 parallel environments, each env's hidden states are swapped in/out of the agent before processing.

### 5.3 DriftWrapper

`envs/drift_wrapper.py` wraps a standard LBF ForagingEnv:

1. On `reset()`: advances the OU process, samples agent composition from the current type-frequency vector, samples food levels, injects both into the inner environment.
2. Within an episode: composition and food are held fixed.
3. Exposes `composition`, `agent_levels`, `food_levels`, `ou_state` properties for logging.

Two food sampling modes:
- **fixed** (primary): Food levels drawn from {2: 0.6, 3: 0.4} regardless of team. Creates capability-task mismatch under drift.
- **coupled** (ablation): Food levels centered on mean agent level. Removes capability confound.

### 5.4 Training Infrastructure

All training uses 16 parallel environments with per-env hidden state management. Training budget: 128,000 episodes (~6.4M environment steps at 50 steps/episode). Epsilon decays from 1.0 to 0.05 over 4.8M steps.

HPC: SLURM cluster with RTX 6000B GPUs for training, any GPU for eval. Conda environment: `drift-aht`.

### 5.5 Metrics

- **IQM (Interquartile Mean)**: Mean of the middle 50% of episode returns. More robust than mean to outlier episodes.
- **Degradation**: Fractional IQM loss relative to stationary baseline: `1 - IQM / IQM_baseline`.
- **Stability region**: Fraction of (sigma, theta) grid points with degradation < 10%.
- **Heatmaps**: Mean return, IQM return, and degradation across the sweep grid, with stability contour overlay.

---

## 6. File Map

| Component | File | Purpose |
|-----------|------|---------|
| GPL agent | `agents/gpl/gpl_agent.py` | Algorithm 5 training, Q-value computation, action selection |
| GPL + inference | `agents/gpl/gpl_agent_inf.py` | Extended agent with aux head + EMA |
| Type inference | `agents/gpl/type_inference.py` | LSTM type embedding (Eq. 7) |
| Agent model | `agents/gpl/agent_model.py` | GNN + MLP teammate action prediction (Eq. 11-13) |
| Joint Q-value | `agents/gpl/joint_action_value.py` | CG Q-network (Eq. 8-10) |
| Auxiliary head | `agents/gpl/auxiliary_head.py` | Level prediction MLP |
| EMA tracker | `drift/ema_tracker.py` | Cross-episode belief context |
| OU process | `drift/ou_process.py` | Simplex-projected OU drift |
| Drift wrapper | `envs/drift_wrapper.py` | Gym wrapper for drift injection |
| Preprocessing | `envs/env_utils.py` | PREPROCESS (Appendix C.1) |
| Training | `experiments/train_gpl.py` | Q1/Q3 training loop |
| Training (inf) | `experiments/train_gpl_inf.py` | Q3-inf training loop |
| Drift eval | `experiments/eval_drift.py` | Q2/Q4/Q4-inf sweep evaluation |
| Confound analysis | `experiments/analyze_capability_confound.py` | Capability confound decomposition |
| Baseline config | `configs/gpl_lbf.yaml` | Q1/Q2 settings |
| Hardened config | `configs/gpl_lbf_q3_hardened.yaml` | Q3_hardened / Q4_hardened settings |
| Inference config | `configs/gpl_lbf_q3_inf.yaml` | Q3-inf/Q4-inf settings |
| Sweep grid | `configs/drift_sweep.yaml` | 10 sigma x 5 theta eval grid |

---

## 7. Expected Narrative Arc (Thesis Structure)

1. **Baseline is robust but vacuously so** (Q1 vs Q2): Under the original Rahman et al. configuration, GPL barely degrades under drift. But this is because full observability + optional cooperation means GPL never needs to infer latent teammate types — the policy works regardless of who the teammates are. This is not a meaningful robustness result.

2. **Hardening exposes fragility** (Q3 vs Q4): When we remove the information shortcuts (partial obs, latent types, mandatory cooperation), GPL must actually rely on its type inference mechanism. Under drift, this mechanism degrades because:
   - Type embeddings were not supervised to encode meaningful teammate structure
   - The LSTM has no cross-episode memory for population trends
   - The coordination graph sees more diverse pairwise interactions (6 pairs vs 3)

3. **Auxiliary inference partially recovers** (Q3-inf vs Q4-inf): The auxiliary level prediction head forces type embeddings to encode meaningful structure (even without observing levels directly), and the EMA tracker provides population context that enables implicit drift detection. The combined remedy should reduce Q4-inf degradation relative to Q4, demonstrating that targeted architectural additions can improve drift robustness.

4. **Capability confound must be controlled for** (confound analysis): Throughout, the capability confound analysis separates genuine non-stationarity effects from task-difficulty effects, ensuring that claimed degradation (or recovery) reflects the agent's adaptive capacity rather than team strength changes.

---

## 8. Future Work (Post-Thesis)

- **Population-conditioned policy**: Replace EMA with a full population encoder that transforms type embedding statistics into a policy-conditioning vector, trained end-to-end. This would generalize the EMA's first-order summary to a learned nonlinear representation of population state.

- **Wolfpack domain**: Extend the analysis to Adhoc-wolfpack-v5, which has genuine openness (agents join/leave mid-episode) and predator-prey dynamics. The PREPROCESS infrastructure is already implemented.

- **Adaptive EMA**: Learn the EMA decay rate alpha as a function of observed type embedding variance, enabling faster adaptation when drift is detected.

- **Non-stationary training**: Train with OU drift active during training (not just evaluation), potentially with curriculum over sigma values, to produce policies that are inherently drift-aware.

- **Comparison with explicit change-point detection**: Test whether explicit drift detection (e.g., CUSUM on type embedding statistics) combined with policy switching outperforms the implicit adaptation of the EMA approach.
