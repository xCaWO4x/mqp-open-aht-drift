# Stress-Testing Open Ad Hoc Teamwork Under Stochastic Population Composition Drift

## 1. Problem Motivation

Multi-agent reinforcement learning has seen substantial algorithmic progress over the past decade, but the field's evaluation practices have not kept pace. Gorsane et al. (2022) conducted a systematic meta-analysis of 75 published MARL papers and found that the dominant pattern is for authors to evaluate on one or two benchmarks --- most frequently SMAC --- with inconsistent reporting of seeds, confidence intervals, and hyperparameter tuning, making cross-paper comparisons largely unreliable. Papoudakis et al. (2021) further demonstrated that algorithm rankings are sensitive to implementation details such as parameter sharing and reward normalization, and that methods showing strong performance on SMAC often fail to maintain that advantage under sparse rewards or different coordination structures. The situation is compounded by benchmark saturation: SMACv2 (Ellis et al., 2023) showed via the open-loop diagnostic that many original SMAC scenarios can achieve non-trivial win rates with policies that ignore observations entirely, meaning years of reported improvements were partly measuring memorization rather than genuine learning.

The net effect is that the field currently lacks both a standardized evaluation protocol and benchmarks that stress-test algorithms along the axes that matter most for real-world deployment --- chief among them, robustness to changes in the composition and nature of the agent population.

## 2. Existing Efforts to Address Evaluation Gaps

Several papers have attempted to broaden the evaluation surface. Papoudakis et al. (2021) introduced EPyMARL along with two sparse-reward environments (Level-Based Foraging, Robotic Warehouse) specifically designed to expose failure modes not visible in SMAC. Gorsane et al. (2022) proposed a standardized performance evaluation protocol built around statistically robust metrics (interquartile mean, performance profiles) and recommended multi-seed, multi-environment evaluation as a minimum standard. The ad hoc teamwork survey by Mirsky et al. (2022) catalogued evaluation practices across the AHT literature, noting the absence of standardized benchmarks and the reliance on environment-specific evaluation.

These contributions meaningfully raised the methodological bar, but they operate within a shared implicit assumption: the agent population is fixed and known at training time, and evaluation measures performance on a predetermined set of compositions. This leaves an entire axis of generalization --- what happens when the population composition itself is non-stationary --- entirely uncharacterized.

## 3. The Dynamic Composition Problem and COPA

The first serious treatment of dynamic team composition in cooperative MARL came from Liu et al. (2021) with COPA (Coach-Player Multi-Agent Reinforcement Learning for Dynamic Team Composition). COPA introduced a hierarchical framework where a globally-informed coach agent coordinates players with partial observations, and demonstrated zero-shot generalization to team compositions not seen during training. The key mechanism is that COPA samples teams from a predefined set of compositions during training, allowing the coach's attention-based strategy distribution to generalize across that discrete set at test time.

COPA represented a genuine advance over prior CTDE methods that assumed fixed homogeneous rosters, and it established the resource collection and rescue environments as natural testbeds for composition generalization. However, COPA's generalization guarantee is implicitly conditioned on the test composition being drawn from the same discrete distribution used during training. The composition space is fixed and fully known at training time. There is no mechanism for handling cases where the generating distribution over compositions evolves continuously --- where agent capabilities, type frequencies, or team structures shift according to some temporal process between or during deployment.

## 4. Ad Hoc Teamwork: The Right Framework, An Incomplete Solution

The ad hoc teamwork literature, surveyed comprehensively by Mirsky et al. (2022) and originating from Stone et al. (2010), addresses a related but distinct problem: training a single agent to collaborate effectively with teammates whose types are unknown and potentially novel.

Rahman et al. (2021; 2023) introduced GPL (Graph-based Policy Learning), which uses GNN-based coordination graphs to handle variable team compositions under both full and partial observability, and formalized the *open* AHT setting where teammates may join or leave the environment during execution. GPL factors the joint action-value function into individual and pairwise terms over a coordination graph (Eqs. 8--10 in Rahman et al. 2023), infers continuous teammate type vectors via an LSTM (Eq. 7), and predicts teammate actions through a relational forward model (Eqs. 11--13). The architecture supports both Q-learning (GPL-Q, Eq. 17) and soft policy iteration (GPL-SPI, Eqs. 18--19) target computation.

ODITS (Gu et al., 2022) extended this by learning continuous latent representations of teammate behavior online under partial observability, removing the assumption that teammates belong to a finite predefined type set. CIAO (Wang et al., 2024) further formalized open AHT using cooperative game theory, grounding GPL's joint Q-value representation in coalitional affinity games and providing theoretical justification for the coordination graph factorization.

Together, these methods establish that composition variability is a tractable and important research axis. However, they share a critical structural assumption: **the process generating team compositions is stationary**. Teammates are drawn from a fixed (if unknown) distribution, and generalization is evaluated on compositions sampled from that same distribution --- possibly with different counts or previously unseen type combinations, but without any temporal structure to how the distribution itself evolves.

## 5. Non-Stationarity in Multi-Agent Systems

The assumption of stationarity is challenged by a small but growing body of work on non-stationary multi-agent settings. Santos et al. (2021) studied ad hoc teamwork with non-stationary teammates whose policies change over time, though their focus was on individual agent policy drift rather than population-level composition drift. Zhang et al. (2023) addressed sudden policy changes in teammates with their Fastap framework, proposing rapid adaptation mechanisms when a teammate's behavior abruptly shifts. Mao et al. (2024) provided theoretical foundations for model-free non-stationary RL with applications to multi-agent settings, establishing regret bounds that depend on the total variation of the non-stationarity.

These works address non-stationarity at the level of individual agent behavior (policy drift) or theoretical regret. None of them model or evaluate the specific phenomenon of *population composition drift* --- where the distribution over agent types in the team evolves according to a stochastic process. The distinction matters because real-world deployments of multi-agent systems rarely encounter truly stationary population distributions: robot fleets undergo gradual hardware heterogenization, human-AI teams change in skill distribution as participants learn or turn over, and autonomous vehicle platoons encounter continuously shifting mixes of vehicle capabilities. The rate and structure of such drift is itself informative for designing robust policies.

## 6. The Gap

No existing benchmark or evaluation protocol characterizes how algorithms degrade as a function of drift in the composition-generating distribution, and no existing method is designed to exploit the temporal structure of that drift for online adaptation.

Existing MARL evaluation frameworks test generalization across a fixed composition space but do not model the dynamics of how that space changes over time. Methods like GPL and ODITS are designed to handle unknown compositions drawn from a stationary distribution, and implicitly assume their inductive biases (GNN generalization, latent type inference) are sufficient for deployment. Whether they remain sufficient when the distribution drifts --- and at what rate of drift they fail --- is an open empirical and theoretical question with direct practical consequences.

## 7. This Work

We propose a principled stress-testing framework for cooperative MARL algorithms under population composition drift, and use it to motivate and evaluate a lightweight drift-aware adaptation mechanism.

### 7.1 Drift Model

We model the composition-generating distribution as evolving according to a stochastic process. Specifically, we parameterize drift via an **Ornstein-Uhlenbeck (OU) process** over agent type frequencies on the probability simplex, with drift rate sigma and mean-reversion strength theta as explicit control parameters:

```
x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * N(0, I)
x_{t+1} = Pi_simplex(x_{t+1})    [Euclidean simplex projection, Duchi et al. 2008]
```

The OU process provides three desirable properties: (1) mean-reversion keeps the process on the simplex with well-defined stationary behavior, (2) the (sigma, theta) parameterization provides independent control over drift magnitude and reversion speed, and (3) the continuous-time formulation naturally models the smooth population changes observed in real deployments.

At each episode boundary, the OU process is advanced and a new team composition is sampled i.i.d. from the current frequency vector x_t. Within episodes, the composition is held fixed.

### 7.2 Contributions

**Contribution 1: A formal evaluation protocol for composition drift robustness.** We characterize algorithm performance as a function of (sigma, theta) and identify a *stability region* --- the set of drift parameters within which a given method maintains performance within epsilon of its stationary-distribution baseline. This evaluation framework is independent of the adaptation method and can be applied to any open AHT algorithm, making it a reusable contribution to MARL benchmarking practice.

**Contribution 2: Empirical characterization of GPL degradation under drift.** We characterize how GPL degrades under drift across two standard environments (Level-Based Foraging, Wolfpack), identifying specific failure modes: whether performance collapse is gradual or sharp, whether it is driven primarily by composition count changes or type frequency shifts, and whether the GNN's structural generalization provides meaningful protection against distributional drift.

**Contribution 3: A drift-aware online adapter.** We propose a lightweight adapter that conditions the existing GPL policy on an inferred drift state, estimated via Bayesian filtering using the OU process dynamics as a prior. The adapter requires no retraining of the base policy --- it operates as a wrapper that updates a belief over current composition distribution parameters from observed agent interactions, and modulates the type inference or action selection accordingly. We evaluate this adapter's recovery behavior within and near the boundary of the stability region.

### 7.3 Environments and Baselines

We evaluate on two environments standard in the open AHT literature:

- **Level-Based Foraging (LBF)** (Papoudakis et al., 2021): Cooperative food collection on an 8x8 grid with level-based coordination constraints. Agents must coordinate based on food levels, making team composition (agent levels) directly relevant to task performance.
- **Wolfpack**: Cooperative prey capture requiring spatial coordination. Team composition affects the ability to form effective capture formations.

Primary baselines:
- **GPL-Q** and **GPL-SPI** (Rahman et al., 2023): Our primary baselines, representing the state-of-the-art in open AHT under stationarity.
- **Random agent**: Lower bound for sanity checking.
- **ODITS** (Gu et al., 2022): Secondary baseline representing online type inference without graph structure (if time permits).

### 7.4 Experimental Protocol

1. **Train GPL** on LBF and Wolfpack under standard (stationary) conditions, verifying reproduction of baseline performance.
2. **Stability region characterization**: Sweep a grid of (sigma, theta) values, running the trained GPL under DriftWrapper for each grid point. Measure interquartile mean (IQM) return, degradation AUC, and recovery speed.
3. **Failure mode analysis**: For drift parameters near and beyond the stability boundary, analyze per-episode returns, composition logs, and type inference quality to identify specific degradation mechanisms.
4. **Adapter evaluation**: Apply the drift-aware adapter to GPL and re-run the stability sweep, measuring whether the stability region expands and by how much.

### 7.5 Implementation Status

The core infrastructure for this work is implemented:

| Component | Status | Location |
|-----------|--------|----------|
| OU process over simplex | Complete | `drift/ou_process.py` |
| DriftWrapper (gym wrapper) | Complete | `envs/drift_wrapper.py` |
| GPL full implementation (Algs. 2--5) | Complete | `agents/gpl/` |
| PREPROCESS for LBF and Wolfpack | Complete | `envs/env_utils.py` |
| Training configs (LBF, Wolfpack) | Complete | `configs/` |
| Pilot degradation experiment | Skeleton | `experiments/pilot_degradation.py` |
| Test suite (66 tests passing) | Complete | `tests/` |

Remaining:
- Training loop and GPL reproduction (Step 13)
- Evaluation protocol and metrics (Steps 14--16)
- Drift-aware adapter design and implementation (Step 18)
- Full experimental runs and analysis (Steps 19--20)

## 8. Relation to Concurrent Work

Moslemi & Lee (2025) independently investigate stability in the context of dynamic team formation in MARL, using Gale-Shapley stable matching to form within-episode agent groups and showing empirically that stability-inducing matching improves generalization to unseen agent counts. While their use of "stability" is suggestive, it refers to the game-theoretic property of matching outcomes (no blocking pairs) rather than to robustness of coordination performance under temporal distributional shift. Our work is orthogonal: we are not proposing a matching algorithm, and our notion of stability is explicitly about maintaining performance across a continuously drifting composition distribution. Their finding that stability-promoting structures improve generalization under compositional variation is consistent with and motivates our focus on drift robustness as a natural next axis of study.

Santos et al. (2021) address non-stationarity in AHT but focus on individual teammate policy changes rather than population-level composition drift. Zhang et al. (2023) handle sudden policy changes with rapid adaptation but do not model the temporal structure of how the composition-generating distribution evolves. Our OU-based drift model fills a distinct niche: continuous, structured, population-level non-stationarity with explicit parameterization enabling systematic evaluation.

---

## References

Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T. (2008). Efficient projections onto the l1-ball for learning in high dimensions. In *Proceedings of the 25th International Conference on Machine Learning (ICML)*, pp. 272--279.

Ellis, B., Cook, J., Moalla, S., Samvelyan, M., Sun, M., Mahajan, A., Foerster, J., and Whiteson, S. (2023). SMACv2: An improved benchmark for cooperative multi-agent reinforcement learning. In *Advances in Neural Information Processing Systems 36 (NeurIPS)*, pp. 37567--37593. arXiv:2212.07489.

Gorsane, R., Mahjoub, O., de Kock, R.J., Dubb, R., Singh, S., and Pretorius, A. (2022). Towards a standardised performance evaluation protocol for cooperative MARL. In *Advances in Neural Information Processing Systems 35 (NeurIPS)*, pp. 5510--5521. arXiv:2209.10485.

Gu, P., Zhao, M., Hao, J., and An, B. (2022). Online ad hoc teamwork under partial observability. In *Proceedings of the 10th International Conference on Learning Representations (ICLR)*.

Liu, B., Liu, Q., Stone, P., Garg, A., Zhu, Y., and Anandkumar, A. (2021). Coach-player multi-agent reinforcement learning for dynamic team composition. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139. arXiv:2105.08692.

Mao, W., Zhang, K., Zhu, R., Simchi-Levi, D., and Basar, T. (2024). Model-free nonstationary reinforcement learning: Near-optimal regret and applications in multiagent reinforcement learning and inventory control. *Management Science*, 71(2):1564--1580.

Mirsky, R., Carlucho, I., Rahman, A., Fosong, E., Macke, W., Sridharan, M., Stone, P., and Albrecht, S.V. (2022). A survey of ad hoc teamwork research. In *Multi-Agent Systems (EUMAS)*, Springer LNCS, pp. 275--293. arXiv:2202.10450.

Moslemi, K. and Lee, C.-G. (2025). Learning bilateral team formation in cooperative multi-agent reinforcement learning. In *CoCoMARL 2025 Workshop*. arXiv:2506.20039.

Papoudakis, G., Christianos, F., Schafer, L., and Albrecht, S.V. (2021). Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks. In *NeurIPS 2021 Track on Datasets and Benchmarks*. arXiv:2006.07869.

Rahman, M.A., Hopner, N., Christianos, F., and Albrecht, S.V. (2021). Towards open ad hoc teamwork using graph-based policy learning. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, pp. 8776--8786.

Rahman, A., Carlucho, I., Hopner, N., and Albrecht, S.V. (2023). A general learning framework for open ad hoc teamwork using graph-based policy learning. *Journal of Machine Learning Research*, 24(298):1--74.

Santos, P.M., Ribeiro, J.G., Sardinha, A., and Melo, F.S. (2021). Ad hoc teamwork in the presence of non-stationary teammates. In *EPIA 2021*, Springer LNCS 12981, pp. 648--660.

Stone, P., Kaminka, G.A., Kraus, S., and Rosenschein, J.S. (2010). Ad hoc autonomous agent teams: Collaboration without pre-coordination. In *Proceedings of the 24th AAAI Conference on Artificial Intelligence*, pp. 1504--1509.

Wang, J., Li, Y., Zhang, Y., Pan, W., and Kaski, S. (2024). Open ad hoc teamwork with cooperative game theory. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235, pp. 50902--50930. arXiv:2402.15259.

Zhang, Z., Yuan, L., Li, L., Xue, K., Jia, C., Guan, C., Qian, C., and Yu, Y. (2023). Fast teammate adaptation in the presence of sudden policy change. In *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 216, pp. 2465--2476. arXiv:2305.05911.
