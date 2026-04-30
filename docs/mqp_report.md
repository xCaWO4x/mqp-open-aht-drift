# Stress-Testing Open Ad Hoc Teamwork Under Stochastic Population Composition Drift

---

# Chapter 1: Introduction

## The Problem

When people work in teams, they constantly adjust to the strengths and weaknesses of whoever is on the team. A nurse covering a shift with an unfamiliar resident adapts differently than when paired with an experienced surgeon. A soccer player receiving the ball reads whether their teammate is sprinting or holding position and chooses a pass accordingly. This ability to cooperate with previously unknown partners, without rehearsal or explicit communication of capabilities, is something humans do naturally but machines find very difficult.

In artificial intelligence, the problem of building an agent that can cooperate effectively with teammates it has never encountered before is called *ad hoc teamwork*. The challenge is practical as well as theoretical. Warehouse robots are deployed alongside human workers whose skills vary from shift to shift. Autonomous vehicles share roads with a continuously changing mix of other vehicles, each with different sensing and driving capabilities. Search-and-rescue drones may need to coordinate with ground robots or human responders whose equipment and training differ from mission to mission. In all these settings, the agent cannot assume it knows who its teammates are or what they can do; it must figure that out on the fly.

Recent research has made real progress on this problem. Algorithms now exist that can infer teammate types from observed behavior, build models of what teammates are likely to do next, and choose actions that complement those predictions. One of the most capable of these algorithms is called GPL, for Graph-based Policy Learning. GPL was designed specifically for *open* ad hoc teamwork, where teammates may join or leave during a task and where the agent must handle variable team sizes and unknown teammate capabilities.

However, nearly all existing work on ad hoc teamwork makes a quiet but important assumption: the mix of teammate types the agent will encounter at deployment time is drawn from the same distribution it experienced during training. In other words, even though the specific teammates on any given task are unknown, the overall population from which they come is assumed to be stable. This assumption rarely holds in practice. Robot fleets undergo hardware upgrades at different rates. Human teams turn over as people learn, retire, or transfer. The population of drivers on a highway changes by time of day, season, and region. The distribution of teammate capabilities is not fixed — it drifts.

This project asks what happens when that assumption is relaxed. Specifically, it investigates two questions. First, does a trained ad hoc teamwork policy degrade when the distribution of teammate types changes over time? Second, does the amount of information the agent can observe about its teammates affect how well it cooperates — and if so, is more information always better?

## Project Objectives

The project had three main objectives.

The first objective was to build a controlled evaluation framework for studying how teammate composition drift affects a trained cooperation policy. This required implementing a mathematical model of how the population changes over time — specifically, a stochastic process that smoothly shifts the relative frequencies of different teammate types — and connecting it to a standard multi-agent environment so that trained policies could be tested under a range of drift conditions.

The second objective was to test whether reducing the agent's access to teammate information would make it more vulnerable to composition drift, and whether adding auxiliary inference mechanisms — modules designed to help the agent recover hidden teammate structure — could restore performance. This involved training multiple variants of GPL under different observation regimes and evaluating each variant across a systematic grid of drift conditions.

The third objective, which emerged as the most important during the course of the project, was to characterize how the amount of teammate information available to the agent affects its performance even in the absence of drift. This led to a sweep over different visibility settings under otherwise identical conditions, producing a curve that relates observability to cooperation quality.

## General Procedure

The project used Level-Based Foraging, a cooperative grid-world environment commonly used in the ad hoc teamwork literature, as its experimental testbed. In this environment, agents of different capability levels must coordinate to collect food items whose difficulty depends on the agents' combined strength. The environment is simple enough to allow systematic experimentation but complex enough that coordination matters: agents must choose which food to pursue, when to wait for help, and how to position themselves relative to teammates.

The agent under study was GPL, trained under two main observation regimes. In the baseline regime, the agent could see the entire grid and directly observe each teammate's capability level. In the restricted regime, the agent's view was limited to a small radius around itself and teammate capability levels were hidden from observation. Between these two extremes, the project trained and evaluated the agent at several intermediate visibility settings.

To model population composition drift, the project implemented an Ornstein-Uhlenbeck process — a standard model for mean-reverting random variation — that controls how the frequencies of different teammate types change from one task episode to the next. By adjusting two parameters of this process, the drift magnitude and the speed of mean reversion, the project could systematically vary the severity of the non-stationarity and map out which conditions the trained policy could tolerate.

Three inference-oriented additions to GPL were also tested: an auxiliary neural network head trained to predict teammate capability from behavioral observations, a cross-episode memory module that tracks the recent population composition, and the combination of both. These were intended to help the agent recover information lost under the restricted observation regime.

All training was performed on a GPU cluster, with each experimental condition trained for 128,000 episodes across 16 parallel environments. Evaluation used greedy action selection over 500 episodes per condition, extended to 2,500 episodes with multiple random seeds for the main comparisons. The project used interquartile mean return as its primary performance metric, chosen for its robustness to the outlier episodes that are common in this type of cooperative task.

## What the Project Found

The project's central finding was unexpected. The original expectation was that hiding teammate information would hurt the agent's performance and that drift would make things worse. The first part of that expectation turned out to be wrong in a surprising way. Across a sweep of visibility settings, the agent performed best not under full observability but under a moderately restricted view. Giving the agent more teammate information did not monotonically improve cooperation; instead, additional visible structure appeared to distract the learning process. This suggests that the observation channel in ad hoc teamwork is not simply a resource to be maximized, but a design variable that can be tuned — and that limiting it can sometimes help.

The drift experiments, by contrast, did not produce the expected degradation. Under the tested conditions, all trained policies remained stable across the full range of drift parameters, including under an alternative food-generation protocol designed to remove a potential confound. This does not prove that GPL is generally robust to composition drift; it indicates that the current experimental setup was not demanding enough to expose a vulnerability. The drift evaluation infrastructure remains a reusable product of the project, ready for application to harder environments.

The auxiliary inference modules also did not improve performance. Diagnostic analysis traced this to a combination of limited information content in the restricted observations and a multi-task interference effect in the shared neural network representation: the dominant training objective did not preserve the teammate-capability signal that the auxiliary module was trying to extract.

## Organization of This Report

The remainder of this report is organized as follows.

The **Literature Review** (Chapter 2) surveys the ad hoc teamwork literature, the treatment of partial observability and non-stationarity in multi-agent reinforcement learning, and the specific gap this project addresses: the absence of systematic evaluation under population composition drift.

The **Hypotheses and Specific Aims** (Chapter 3) states the three pre-registered hypotheses — about drift-induced degradation, an observability threshold for teammate inference, and an interaction between drift and observability — and defines the concrete experimental aims that test them.

The **Methodology** (Chapter 4) describes the experimental procedure in detail: the environment setup, observation regimes, drift model, model variants, training protocol, evaluation protocol, and metrics. It is written as a tutorial for reproducing the project.

The **Results** (Chapter 5) presents the measured outcomes — stationary baselines, drift-sweep grids, sight-sweep curves, and auxiliary-head diagnostics — with tables and confidence intervals but without interpretation.

The **Analysis and Discussion** (Chapter 6) interprets those results against the Specific Aims and hypotheses, explains the mechanisms behind the observed patterns, discusses the assumptions that shape the interpretation, and identifies the limitations of the current evidence.

The **Conclusion** (Chapter 7) draws the global lessons of the project: that observation design is the strongest empirical contribution, that drift is a non-result under the current protocol, and that the inference remedies failed for identifiable and instructive reasons.

The **Recommendations** (Chapter 8) lays out prioritized next steps for continuing this work into a publication, organized into three tiers by importance. The most critical recommendations are learned observation masking, multi-environment replication, and a denser multi-seed sight sweep. Together, these would convert the thesis finding into a general, algorithmically grounded contribution to the open ad hoc teamwork literature.

---

# Chapter 2: Literature Review

This review orients the reader toward the MQP's actual design: **GPL** under **population composition drift**, with a controlled **information nerf** that makes teammate capability only partially observable. The narrative keeps two difficulties—**limited sensing of teammates** and **non-stationarity of the composition-generating process**—on separate axes so neither is confused for the other.

---

## 1. Ad Hoc Teamwork as the Common Problem

Stone and colleagues formalized *ad hoc autonomous agent teams*: a single learner must collaborate with previously unknown teammates, without pre-coordination, on tasks where every agent can contribute.[^stone] Subsequent work expanded AHT to open settings (variable team size, joining and leaving, unknown teammate policies) and surveyed a large body of methods and benchmarks.[^mirsky] That survey usefully stresses evaluation diversity; it does not, by itself, prescribe how policies should behave when the **distribution** from which teammates are drawn **changes over time**—the gap this project stresses.

---

## 2. Two Axes of Difficulty (Kept Distinct)

Many MARL discussions collapse "hard multi-agent settings" into one bucket. For this project, two structurally different axes matter.

| Axis | What it refers to | How the literature usually frames it | How this MQP instantiates it |
|------|-------------------|--------------------------------------|-------------------------------|
| **A. Partial observability of teammate structure** | The focal agent's observations do not fully identify teammates' capabilities or roles; inference must bridge the gap. | Partially observable (multi-agent) decision problems; centralized training with decentralized execution (CTDE) when extra information exists only at training time; latent-type or latent-policy models in AHT. | **Information nerf:** reduced sight radius and hiding teammate level in Level-Based Foraging (LBF), so behavioral inference (e.g., GPL's LSTM type vectors) becomes load-bearing. This axis is *directly* about what the learner can see. |
| **B. Non-stationarity of the composition process** | The rule that draws *which* teammates appear **across episodes** is not fixed; the learner may face a shifting mixture over types or policies even when within-episode dynamics are Markov given the drawn team. | Distinct threads: heterogeneous teams (agents differ), **open** composition (sets or types vary), **stationary** sampling (i.i.d. from a fixed mixture), versus **drift** (the mixture or population statistics evolve). | **Population drift:** an Ornstein–Uhlenbeck process on the type-frequency simplex advances at episode boundaries; compositions are sampled from the current mixture. |

**Framing discipline (axis B).** Heterogeneity—agents differing in level, skill, or objective—is almost always present in AHT and is a *prerequisite* for composition to matter. It is **not** the same object as drift. One can hold heterogeneity fixed (several discrete types) while making the **generating process** over those types **stationary** (standard open-AHT evaluation: random teammates i.i.d. from a fixed distribution[^rahman-jmlr]). Conversely, drift in this MQP is **not** "more heterogeneity" in the abstract; it is **temporal structure on the mixture** over types that were already in the training support. The motivating deployments (rotating personnel, shifting skill mixes, fleets replaced over quarters) are stories about **how often** each type appears, not necessarily about inventing wholly new types. That distinction matters when connecting to prior work on *open-set* generalization (novel types never seen at training), which this project explicitly does not emphasize.

**Framing discipline (axis A).** Partial observability here is **operational**: we restrict sensing (sight, level visibility) so that the same GPL machinery must infer latent structure from trajectories. That is aligned with POMDP-style difficulty but **does not** by itself implement drift; it sharpens the need for type inference so that downstream questions about robustness are non-vacuous.

---

## 3. Axis A—Partial Observability, Latent Teammates, and GPL-Class Methods

GPL (Graph-based Policy Learning) represents the state of the art for **open** AHT under variable composition: an LSTM produces continuous type embeddings, a relational model predicts teammate actions, and a coordination-graph Q-factor selects the learner's action—supporting both full and reduced observability depending on environment features.[^rahman-icml][^rahman-jmlr] ODITS removes reliance on a small finite type set by learning latent teammate structure **online** under partial observability—closer to the "decode the team from what you see" problem this MQP foregrounds when levels are hidden.[^gu] CIAO reframes open AHT with cooperative game theory and offers theoretical grounding for graph-structured value decompositions akin to GPL's coordination graph.[^wang-ciao]

Privileged information at training time (CTDE) appears across MARL; in AHT it supports auxiliary objectives that shape teammate models without requiring privileged fields at execution. ODITS's decoding-style losses are one example; our later auxiliary-level head (see research document §4) sits in the same family—predicting a discrete capability from embeddings—rather than claiming a new representational paradigm.

**Fast teammate adaptation (Fastap).** Zhang and colleagues study **abrupt** change in a teammate's *policy* and rapid adaptation—non-stationarity at the behavioral level within the AHT framing.[^zhang-fastap] That complements, and should not be conflated with, **mixture drift** over types across episodes: Fastap targets a different time scale and a different latent (policy identity versus population frequency).

---

## 4. Axis B—Composition, Populations, and Stationarity in Prior Work

**Generalization across compositions, still stationary.** COPA uses a coach–player hierarchy and training over a predefined set of team layouts to zero-shot generalize to held-out layouts from the **same** discrete family.[^liu-copa] The emphasis is spatial or roster structure, not a continuously drifting **frequency** over types through time.

**Population- and policy-aware extensions.** Recent N-agent ad hoc teamwork (NAHT) and the POAM algorithm broaden who is controlled and emphasize learning teammate behavior representations for out-of-distribution teammates at evaluation—again advancing composition flexibility, with evaluation still typically embedded in **stationary** test protocols unless authors add explicit temporal drift.[^wang-poam]

**Other non-stationary AHT/MARL threads.** Santos and colleagues study AHT when individual teammates are **non-stationary** in the sense of changing behavior over time.[^santos] Mao and colleagues give model-free non-stationary RL regret theory with multi-agent applications—valuable for abstract non-stationarity, not a direct model of simplex mixture drift.[^mao] These works justify treating "stationary i.i.d. teammates" as a special case rather than the only realistic case; they do not replace a **parameterized drift model** over population mixtures for the question we ask here.

---

## 5. Synthesis: Why Both Axes Appear in This Project

GPL's published LBF evaluation uses observations rich enough that type inference is largely bypassable; under that regime, **axis B** (drift over mixtures) barely stresses the algorithm because **axis A** never forces the policy to depend on inferred types. The MQP therefore **orthogonalizes** the story: (1) an information-nerfed regime makes **axis A** real; (2) OU drift on the mixture implements **axis B** as a controlled, smooth, temporally correlated departure from episode-i.i.d. stationarity. Neither axis "substitutes" for the other: drift does not create partial observability, and hiding levels does not model a drifting population—**together** they ask whether graph-based open-AHT learning remains useful when both difficulties apply.

---

## Notes

[^stone]: Peter Stone, Gal A. Kaminka, Sarit Kraus, and Jeffrey S. Rosenschein, "Ad Hoc Autonomous Agent Teams: Collaboration without Pre-Coordination," in *Proceedings of the 24th AAAI Conference on Artificial Intelligence* (2010), 1504–9.

[^mirsky]: Reuth Mirsky et al., "A Survey of Ad Hoc Teamwork Research," in *Multi-Agent Systems: EUMAS 2022*, Springer LNCS (2022), 275–93, arXiv:2202.10450.

[^rahman-icml]: Md. Ashiqur Rahman et al., "Towards Open Ad Hoc Teamwork Using Graph-Based Policy Learning," in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139 (2021), 8776–86.

[^rahman-jmlr]: Arrasy Rahman et al., "A General Learning Framework for Open Ad Hoc Teamwork Using Graph-Based Policy Learning," *Journal of Machine Learning Research* 24, no. 298 (2023): 1–74.

[^gu]: Peihao Gu et al., "Online Ad Hoc Teamwork under Partial Observability," in *Proceedings of the International Conference on Learning Representations (ICLR)* (2022).

[^wang-ciao]: Jiachen Wang et al., "Open Ad Hoc Teamwork with Cooperative Game Theory," in *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235 (2024), 50902–30, arXiv:2402.15259.

[^liu-copa]: Boyuan Liu et al., "Coach-Player Multi-Agent Reinforcement Learning for Dynamic Team Composition," in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139 (2021), arXiv:2105.08692.

[^wang-poam]: Caroline Wang et al., "N-Agent Ad Hoc Teamwork," *Advances in Neural Information Processing Systems* 37 (NeurIPS 2024), arXiv:2404.10740.

[^santos]: Pedro M. Santos et al., "Ad Hoc Teamwork in the Presence of Non-Stationary Teammates," in *Progress in Artificial Intelligence: EPIA 2021*, Springer LNCS 12981 (2021), 648–60.

[^zhang-fastap]: Ziqian Zhang et al., "Fast Teammate Adaptation in the Presence of Sudden Policy Change," in *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 216 (2023), 2465–76, arXiv:2305.05911.

[^mao]: Weichao Mao et al., "Model-Free Nonstationary Reinforcement Learning: Near-Optimal Regret and Applications in Multiagent Reinforcement Learning and Inventory Control," *Management Science* 71, no. 2 (2024): 1564–80.

---

## Bibliography

Gu, Peihao, Minguk Zhao, Jianye Hao, and Bo An. "Online Ad Hoc Teamwork under Partial Observability." In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2022.

Liu, Boyuan, Qi Liu, Peter Stone, Anima Garg, Yuke Zhu, and Anima Anandkumar. "Coach-Player Multi-Agent Reinforcement Learning for Dynamic Team Composition." In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 2021. arXiv:2105.08692.

Mao, Weichao, Kaiqing Zhang, Ruihao Zhu, David Simchi-Levi, and Tamer Başar. "Model-Free Nonstationary Reinforcement Learning: Near-Optimal Regret and Applications in Multiagent Reinforcement Learning and Inventory Control." *Management Science* 71, no. 2 (2024): 1564–80.

Mirsky, Reuth, Ignacio Carlucho, Arrasy Rahman, Elise Fosong, Wendy Macke, Mohan Sridharan, Peter Stone, and Stefano V. Albrecht. "A Survey of Ad Hoc Teamwork Research." In *Multi-Agent Systems: EUMAS 2022*, edited by Nils Bulling and Amro Najjar, 275–93. Cham: Springer, 2022. arXiv:2202.10450.

Rahman, Arrasy, Ignacio Carlucho, Niklas Höpner, and Stefano V. Albrecht. "A General Learning Framework for Open Ad Hoc Teamwork Using Graph-Based Policy Learning." *Journal of Machine Learning Research* 24, no. 298 (2023): 1–74.

Rahman, Md. Ashiqur Rahman, Niklas Höpner, Filippos Christianos, and Stefano V. Albrecht. "Towards Open Ad Hoc Teamwork Using Graph-Based Policy Learning." In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 8776–86, 2021.

Santos, Pedro M., João G. Ribeiro, Ana Sardinha, and Francisco S. Melo. "Ad Hoc Teamwork in the Presence of Non-Stationary Teammates." In *Progress in Artificial Intelligence*, edited by Luís Soares Barbosa and Luis Farinas del Cerro, 648–60. Cham: Springer, 2021.

Stone, Peter, Gal A. Kaminka, Sarit Kraus, and Jeffrey S. Rosenschein. "Ad Hoc Autonomous Agent Teams: Collaboration without Pre-Coordination." In *Proceedings of the 24th AAAI Conference on Artificial Intelligence*, 1504–9. AAAI Press, 2010.

Wang, Caroline, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone. "N-Agent Ad Hoc Teamwork." In *Advances in Neural Information Processing Systems* 37, 2024. arXiv:2404.10740.

Wang, Jiachen, Yanchen Li, Yifeng Zhang, Wei Pan, and Samuel Kaski. "Open Ad Hoc Teamwork with Cooperative Game Theory." In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 50902–30, 2024. arXiv:2402.15259.

Zhang, Ziqian, Lei Yuan, Lu Li, Kai Xue, Chenhao Jia, Cong Guan, Chun Qian, and Yang Yu. "Fast Teammate Adaptation in the Presence of Sudden Policy Change." In *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence*, PMLR 216, 2465–76, 2023. arXiv:2305.05911.

---

# Chapter 3: Hypotheses and Specific Aims

This chapter states the hypotheses that drive the project and the specific aims that operationalise them. The hypotheses are **pre-registered**: each is a one-sentence prediction about how the learned GPL policy should behave along one or both of the two axes identified in the literature review (axis A, *partial observability of teammate structure*; axis B, *non-stationarity of the composition-generating process*). Each hypothesis is tied to a concrete sweep that either supports or rules it out. The hypotheses are educated guesses in the sense of Tien and Tien (2010) — if the data contradict them, the result is still informative, and the sweeps were designed so that either outcome yields a publishable empirical characterisation of GPL under real-world-like deployment stress.

---

## 1. Motivating Observations

Two facts about the existing open ad hoc teamwork (AHT) literature shape the hypotheses below.

**Observation 1 (from the literature review, §2).** The field has treated axis A (observability) and axis B (non-stationarity) in largely separate threads: GPL and ODITS address latent-type inference under partial observability against a **stationary** teammate distribution[^rahman-jmlr][^gu]; Santos et al. and Zhang et al. address non-stationary teammate behaviour but with **full** or near-full observability of teammate identity[^santos][^zhang-fastap]. No prior work, to our knowledge, evaluates an open-AHT policy on the **joint** axis — partial observability combined with a parameterised drift on the composition-generating distribution. The *a priori* expectation from first principles is that the two axes should not be independent: if the policy must already infer teammate types from behaviour under observability constraints, then changing the distribution from which those types are drawn should erode inference quality in a way that full-observability drift would not.

**Observation 2 (from the pilot runs, §4 of the research document).** GPL's published LBF evaluation uses a nearly-full observation (sight radius = 8 on an 8×8 grid, teammate level directly in the observation vector), under which the LSTM type-inference head is only weakly load-bearing. Restricting the observation to the "information-nerfed" setting (sight = 3, teammate level hidden) produces a policy that trains to a lower asymptote and shows a qualitatively different failure signature under drift. This, combined with the observation that the auxiliary level-prediction head failed to reduce cross-entropy below `ln 3` at sight = 3 (research doc §4.7), suggests that observability is not a smooth knob but may carry a threshold below which trajectory-based type inference is **informationally infeasible** regardless of algorithmic fix.

These two observations jointly motivate treating observability and drift as co-axes with potentially interacting effects, and motivate the hypothesis that performance along each axis is not monotone-but-linear but exhibits a **phase-transition-like** collapse at some axis-specific threshold.

---

## 2. Hypotheses

Three hypotheses, each stated in a single testable sentence, with the rationale and the concrete sweep that verifies or falsifies it.

### H1 — Drift degradation (axis B alone)

> **H1.** Holding observability fixed, a stationary-trained GPL policy's stationary-baseline-normalised IQM return is a monotone non-increasing function of drift magnitude (σ) and of slowness-of-reversion (1/θ), and collapses to near-chance performance beyond a finite drift budget.

**Rationale.** GPL's LSTM produces type embeddings conditioned on a single episode's trajectory; it has no cross-episode memory, and it was trained against a distribution of teammate mixtures drawn i.i.d. from a *stationary* population prior. When the OU drift advances the simplex at episode boundaries, the distribution of mixtures encountered at evaluation departs from the training support, and — since drift carries no within-episode signal — the LSTM cannot close the gap. We therefore predict monotone degradation as the drift knobs are turned up, bounded below by the random-teammate lower bound and bounded above by the stationary-baseline ceiling. The "collapse beyond a finite budget" clause is the non-trivial part: it commits to the claim that degradation is not merely gradual but reaches a regime where the policy is *qualitatively* decoupled from the teammate distribution, not just noisier.

**Test.** The 50-cell (σ, θ) drift grid of Q2 (full observability, sight = 8) and Q4_rw (information nerf, sight = 3). For each cell we compute IQM return over 500 episodes × 5 seeds, normalise by the same-agent stationary (σ = 0) baseline, and fit a stability region at a 10 % degradation threshold. H1 is supported if the stability region has a finite boundary that stays finite as the grid is extended; it is falsified if IQM return is flat in (σ, θ) out to the maximum drift the OU process supports on the simplex.

### H2 — Observability threshold for teammate inference (axis A alone)

> **H2.** Under the information-nerfed regime (`observe_agent_levels = false`), there exists a sight-radius threshold `s*` below which any behavioural auxiliary inference signal (e.g. a level-prediction head sharing GPL's type encoder) cannot reduce cross-entropy meaningfully below the uniform-prior floor, and above which the same signal becomes extractable and translates into stationary return gains.

**Rationale.** At sight = 3 on an 8×8 grid with episodes of length 50, the mutual information `I(trajectory ; teammate level)` is bounded above by how often a teammate is visible and by how many level-discriminating actions they take while visible. The research document's §4.7 short-checks show that cross-entropy of the auxiliary head stays within thousandths of `ln 3` across the entire training run at sight = 3, with scaling `aux_weight` from 0.1 to 2.0 having essentially no effect — consistent with a hard information floor rather than an optimisation pathology. Since the same pipeline with `observe_agent_levels = true` does reduce cross-entropy below `ln 3` (to ≈0.90), the plumbing is correct and the floor at sight = 3 is environmental. As sight radius grows, the teammate becomes more frequently visible and the informative-action budget grows; we therefore predict a threshold `s* > 3` at which the auxiliary head starts to learn and a corresponding improvement in stationary IQM.

**Test.** The Q3-inf-aux **sight sweep** at `sight ∈ {4, 5, 6, 7}`, holding all other hyperparameters identical to the `sight = 3` Q3-inf-aux baseline (single seed, same 128 k-episode budget, same `aux_weight = 0.1`). For each sight point we record (a) the training-time auxiliary cross-entropy trajectory and (b) the 500-episode stationary greedy IQM return. H2 is supported if mean CE drops below `ln 3 − ε` at some sight `s ≤ 7` and the corresponding IQM exceeds the `sight = 3` Q3_rw baseline; H2 is falsified if CE stays pinned near `ln 3` and IQM is flat (or worse, lower) across the sweep. A graceful-slope outcome — CE and IQM moving gradually with no clear break — is consistent with H2's existence claim but not its phase-transition phrasing, and would itself be a reportable negative result against the stronger reading.

### H3 — Axis A and axis B interact (the main hypothesis)

> **H3.** The drift-induced collapse of H1 occurs at *smaller* drift budgets under the information nerf than under full observability, i.e. the two-axis (sight, drift) surface is sub-additive in the sense that the policy's stability region shrinks faster along the drift axis as observability is reduced.

**Rationale.** The literature keeps axes A and B separate, but mechanistically they are coupled through the type-inference bottleneck: under full observability, GPL can rely on the level feature as a shortcut for teammate identity and the LSTM's role is secondary; under the information nerf, the LSTM is *the* mechanism that connects observed behaviour to Q-values. Any drift-induced degradation of the type-inference signal should therefore be amplified when the LSTM is the binding constraint rather than a redundant supplement. Practically, we expect the Q4_rw (`sight = 3`) drift grid to show a smaller stability region than the Q2 (`sight = 8`) drift grid evaluated against the same baseline-normalised threshold, and for that gap to narrow as sight is relaxed.

**Test.** A two-condition comparison on the 50-cell drift grid: Q2 (sight = 8, levels on) versus Q4_rw (sight = 3, levels off), each against its own stationary-baseline-normalised degradation map. H3 predicts the sight = 3 stability region is a strict subset of the sight = 8 stability region at the 10 % degradation threshold. A stronger version of H3, pursued as future work (§3, Aim 3 below), replaces the two-condition comparison with a continuous sweep over sight and asks whether the stability-region boundary slides monotonically with sight; this requires the sight sweep's trained checkpoints to be re-evaluated under the full drift grid, not just under the stationary greedy protocol.

### What "collapse" means

For both H1 and H3 the word **collapse** is used in a technical sense: we will say the policy has collapsed at a (σ, θ) cell when its IQM return at that cell is statistically indistinguishable (bootstrap 95 % CI) from a trained-but-random-teammate lower bound obtained by evaluating the stationary checkpoint against uniformly sampled team compositions. This is a stricter criterion than "degraded below the 10 % threshold" and is reserved for claims about *regime change* rather than *gradual erosion*. The two thresholds together give the drift surface a three-region partition — stable, degraded, collapsed — that is used in the results chapter to report findings.

---

## 3. Specific Aims

Each specific aim is one sweep or one comparison, with a single sentence describing what it produces and which hypothesis it tests.

**Aim 1. Stationary-baseline reproduction under both observability regimes.** Train GPL-Q to convergence on the two observation settings — Q1 (sight = 8, levels observed) and Q3_rw (sight = 3, levels hidden) — and confirm that both reach a stable greedy IQM on the stationary (σ = 0) evaluation, establishing the *baselines against which drift degradation is measured* in Aim 2. This aim does not itself test a hypothesis but is a prerequisite for everything that follows.

**Aim 2. Drift stability-region characterisation at both observability regimes (tests H1 and H3).** Sweep the 50-cell (σ, θ) grid for each of Q2 (Q1 checkpoint) and Q4_rw (Q3_rw checkpoint), 500 episodes × 5 seeds per cell, computing IQM return, baseline-normalised degradation, and the random-teammate collapse threshold. Aim 2 produces the two heatmaps whose comparison is H3's primary test; each heatmap in isolation tests H1. *Current status:* the completed drift sweeps are not diagnostically useful in the present setup. Under fixed food the maps are largely flat, and a coupled-food re-run of Q2 also shows no meaningful degradation boundary, so this aim is presently secondary to the stationary observability results.

**Aim 3. Observability-threshold characterisation via the sight sweep (tests H2).** Train Q3-inf-aux at `sight ∈ {4, 5, 6, 7}` under otherwise-identical settings to the `sight = 3` Q3-inf-aux baseline, and evaluate each resulting checkpoint under the stationary greedy protocol (500 episodes, σ = 0) to produce an *IQM-vs-sight* curve plus the matching auxiliary-CE trajectory. Aim 3 locates the empirical threshold `s*` of H2 (or rules out a sharp threshold at the sight resolutions we can afford). *Current status:* the stationary sight sweep is now the central empirical result of the project. A multi-seed Q3_rw sweep and the currently available single-seed Q3-inf-aux sweep both reject a monotone "more sight is better" story and instead indicate a non-monotone curve with the strongest performance around `sight = 4`.

**Aim 4 (future work, conditional on Aim 3 outcome; tests the strong form of H3).** If Aim 3 identifies a threshold, re-run the 50-cell drift grid on each sight-sweep checkpoint to produce a (sight × σ × θ) volume, allowing the stability-region boundary to be tracked as a function of sight. This converts the two-point H3 comparison of Aim 2 into a continuous learnability surface and is the single strongest reviewer-facing extension identified in §8 of the research document. It is out of scope for the present project's compute budget but is flagged here because its design follows directly from the present hypotheses.

---

## 4. What Would Falsify Each Hypothesis

The hypotheses are useful only to the extent they could have been wrong. The following outcomes, each of which is fully reachable from the sweeps listed above, would rule out their respective hypothesis:

- **H1 falsified** by a Q2 or Q4_rw drift map that is flat in (σ, θ) out to the extreme corners of the grid — degradation never crosses the 10 % threshold.
- **H2 falsified** by a sight sweep in which auxiliary cross-entropy remains within ε of `ln 3` at *every* sight point (no threshold ≤ 7 detectable on an 8×8 grid), or by one in which CE drops but stationary IQM does not track.
- **H3 falsified** by a Q4_rw stability region that equals or contains the Q2 region — i.e. the information nerf does not make the policy more drift-fragile.
- **Phase-transition strong form falsified** (H1/H2 weaker versions preserved) by smooth, featureless degradation curves with no visible regime change — a result worth reporting as a negative finding against the phase-transition framing in §2, even though the existence-of-degradation claim would survive.

Each of the first three is directly readable from the heatmaps and IQM-vs-sight plots produced by Aims 2 and 3; the fourth is readable from the shape of those curves and from the CE-vs-episode training logs.

---

## 5. Current Status After Phase-One Sweeps

The hypotheses above are kept as the original pre-registered predictions. The current empirical status, however, is already informative:

- **H1 is not supported in the present setup.** The current Q2 and Q4_rw drift maps under fixed food are largely flat, and a coupled-food re-run of Q2 likewise shows no meaningful degradation boundary, so the drift axis is not the main result of the present draft.
- **H2 is falsified in its strong threshold form and replaced by a stronger non-monotonicity result.** In the completed stationary sweeps, the best RW performance occurs at an intermediate sight level rather than at maximal sight, and the full-observability Q1 baseline underperforms every measured RW sight point.
- **H3 is not supported in the present setup.** Since Q4_rw does not clearly shrink the stability region relative to Q2 under the completed drift sweeps, the anticipated drift × observability interaction is not yet a convincing claim.
- **The project's strongest phase-one contribution is therefore on observation design.** The current evidence supports treating observability not as a monotone resource but as a tunable information bottleneck, with moderate restriction improving stationary performance.

---

[^rahman-jmlr]: Arrasy Rahman et al., "A General Learning Framework for Open Ad Hoc Teamwork Using Graph-Based Policy Learning," *Journal of Machine Learning Research* 24, no. 298 (2023): 1–74.

[^gu]: Peihao Gu et al., "Online Ad Hoc Teamwork under Partial Observability," in *Proceedings of the International Conference on Learning Representations (ICLR)* (2022).

[^santos]: Pedro M. Santos et al., "Ad Hoc Teamwork in the Presence of Non-Stationary Teammates," in *Progress in Artificial Intelligence: EPIA 2021*, Springer LNCS 12981 (2021), 648–60.

[^zhang-fastap]: Ziqian Zhang et al., "Fast Teammate Adaptation in the Presence of Sudden Policy Change," in *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 216 (2023), 2465–76, arXiv:2305.05911.

---

# Chapter 4: Methodology

This chapter explains how the project was conducted and how the Specific Aims were operationalized in code, configuration, and evaluation. The goal is to make the work reproducible for a new researcher joining the project. Wherever possible, the description below names the exact scripts, configuration files, and result directories used in the experiments.

The methodology follows the final phase-one framing of the project:

- the **stationary observability axis** is the primary empirical focus;
- the **drift axis** was implemented and evaluated carefully, but under the present LBF setup it did not produce a meaningful degradation boundary and is therefore reported as a non-diagnostic axis rather than as the thesis-facing central result.

## 1. Methodological Overview

The project asked how GPL behaves when two deployment stresses are introduced:

1. **Restricted observability of teammate structure**: the learner sees less information about teammates, especially their levels.
2. **Non-stationary teammate composition**: the population distribution drifts across episodes according to an Ornstein-Uhlenbeck (OU) process.

To study these axes systematically, the project used a quadrant design plus inference ablations:

|  | Stationary (no drift) | Drift |
|---|---|---|
| **Baseline** (full observability) | **Q1** | **Q2** |
| **RW baseline** (information nerfs only) | **Q3_rw** | **Q4_rw** |
| **RW + aux head only** | **Q3-inf-aux** | **Q4-inf-aux** |
| **RW + EMA only** | **Q3-inf-ema** | **Q4-inf-ema** |
| **RW + aux head + EMA** | **Q3-inf** | **Q4-inf** |

The logic of the sequence was:

1. Reproduce a GPL baseline under the original Rahman-style LBF setup.
2. Introduce an information-nerfed setting that removes the direct teammate-level shortcut.
3. Add inference modules intended to recover performance under reduced observability.
4. Evaluate the trained checkpoints both under stationary greedy evaluation and under OU-driven drift.
5. Extend the stationary comparison with a **sight sweep** to measure how performance changes as visibility is relaxed from the most restricted setting.

This chapter documents each of those steps.

## 2. Codebase and Software Stack

The implementation is organized around a small number of core modules:

| Component | File | Role |
|---|---|---|
| Baseline training loop | `experiments/train_gpl.py` | Trains GPL under stationary composition |
| Inference training loop | `experiments/train_gpl_inf.py` | Trains GPL with auxiliary head and/or EMA tracker |
| Drift evaluation | `experiments/eval_drift.py` | Evaluates trained checkpoints at a single drift point or across the full drift grid |
| Drift wrapper | `envs/drift_wrapper.py` | Injects OU-driven teammate composition drift and food sampling into LBF |
| Observation preprocessing | `envs/env_utils.py` | Implements PREPROCESS for GPL inputs, including the LBF observation parser |
| Baseline GPL agent | `agents/gpl/gpl_agent.py` | Main GPL implementation |
| Inference GPL agent | `agents/gpl/gpl_agent_inf.py` | GPL plus auxiliary level head and EMA context |

The Level-Based Foraging environment is imported from the installed `lbforaging` package, but the project does **not** use `gym.make()` for the experimental LBF runs. Instead, it constructs `ForagingEnv` directly in code so that all observation, level, and reward behavior is explicit and reproducible.

The project was run on a SLURM-managed GPU cluster. Training and evaluation scripts are stored in `scripts/slurm/`. The default conda environment is `drift-aht`, but the scripts also support `MQP_CONDA_ENV` overrides and detect both `~/miniconda3` and `~/anaconda3`.

## 3. Specific Aims and How They Were Met

The methodology was organized around the Specific Aims in the Hypotheses chapter.

### Aim 1: Stationary baselines under both observability regimes

Aim 1 required training and validating:

- **Q1**: the baseline GPL configuration with full observability
- **Q3_rw**: the information-nerfed GPL configuration

This was implemented with:

- `experiments/train_gpl.py`
- `configs/gpl_lbf.yaml` for Q1
- `configs/gpl_lbf_q3_rw.yaml` for Q3_rw
- `scripts/slurm/q1_train.slurm`
- `scripts/slurm/q3_rw_train.slurm`

The stationary greedy evaluation protocol was then run on the trained checkpoints using `experiments/eval_drift.py` in single-point mode with:

- `sigma = 0.0`
- `theta = 0.15`
- `n_episodes = 500`
- greedy policy (`epsilon = 0`)

For phase one, this stationary evaluation was later extended to **5 seeds x 500 episodes** for Q1 and for the Q3_rw sight sweep to reduce sampling noise.

### Aim 2: Drift stability-region characterization

Aim 2 required evaluating trained Q1 and Q3_rw checkpoints under a full drift grid:

- `Q1 -> Q2`
- `Q3_rw -> Q4_rw`

This was implemented with:

- `experiments/eval_drift.py --sweep`
- `configs/drift_sweep.yaml`
- `scripts/slurm/q2_drift_eval.slurm`
- `scripts/slurm/q4_rw_drift_eval.slurm`

The same code path was also used for:

- `Q4-inf-aux`
- `Q4-inf-ema`
- `Q4-inf`

using the corresponding inference checkpoints and configs.

The drift axis was completed faithfully, including a later `Q2` coupled-food re-run (`scripts/slurm/q2_drift_eval_coupled.slurm`), but in the present setup these sweeps remained largely flat or improving and therefore did not become the main empirical claim.

### Aim 3: Observability characterization via the sight sweep

Aim 3 asked whether relaxing visibility from the most restricted RW setting would improve teammate inference and stationary control.

This was implemented in two complementary ways:

1. **Q3-inf-aux sight sweep**
   - configs: `configs/gpl_lbf_q3_inf_aux_sight{4,5,6,7}.yaml`
   - training scripts: `scripts/slurm/q3_inf_aux_sight{4,5,6,7}_train.slurm`
   - eval scripts: `scripts/slurm/q3_inf_aux_sight{4,5,6,7}_greedy_eval.slurm`

2. **Q3_rw sight sweep**
   - configs: `configs/gpl_lbf_q3_rw_sight{4,5,6,7}.yaml`
   - training scripts: `scripts/slurm/q3_rw_sight{4,5,6,7}_train.slurm`
   - eval scripts: `scripts/slurm/q3_rw_sight{4,5,6,7}_greedy_eval.slurm`

The RW sight sweep was later extended with multi-seed stationary evaluation using:

- `scripts/slurm/q3_rw_multiseed_greedy_eval.slurm`
- `scripts/slurm/submit_rw_multiseed_eval.sh`
- `scripts/aggregate_rw_sight_sweep.py`

This aim became the central empirical axis of phase one.

## 4. Environment and Task Definition

### 4.1 Level-Based Foraging

All experiments use an 8x8 Level-Based Foraging environment with:

- `n_agents = 3`
- `n_food = 3`
- `max_steps = 50`
- `K = 3` agent types, mapped directly to LBF levels `{1, 2, 3}`

The baseline and RW settings keep the same agent count and cooperation structure so that the main manipulated variable is information availability, not task topology.

### 4.2 Observation format

The LBF observations are ego-centric and encoded in the underlying environment as:

```text
[food_0(y, x, level), food_1(y, x, level), ..., self(...), other_0(...), ...]
```

The important implementation detail is that **food features come first, then agent features**. This is handled in `envs/env_utils.py::preprocess_lbf()`.

The preprocessing logic reconstructs a global state and then applies GPL's PREPROCESS decomposition:

- `x_j`: agent-specific features for agent `j`
- `u`: shared food features
- `B_j = [x_j ; u]`

This means the type-inference LSTM always sees a per-agent input vector consisting of that agent's own features plus shared environment context.

### 4.3 Observability conditions

The project uses two main observation regimes.

#### Baseline full-observability regime (Q1/Q2)

Defined in `configs/gpl_lbf.yaml`:

- `sight = 8` on an 8x8 grid (effectively full-grid visibility)
- `observe_agent_levels = true`

In this regime:

- per-agent features: `(y, x, level)` -> 3 features
- shared food features: `3 * n_food = 9`
- total `obs_dim = 12`

#### RW information-nerfed regime (Q3_rw/Q4_rw and inference variants)

Defined in `configs/gpl_lbf_q3_rw.yaml` and inherited by the inference configs:

- `sight = 3`
- `observe_agent_levels = false`

In this regime:

- per-agent features: `(y, x)` -> 2 features
- shared food features remain 9
- total `obs_dim = 11`

The only intentional difference from Q1 is the restriction of teammate information, not a change in action space, horizon, or team size.

### 4.4 Food sampling

Two food-sampling modes are implemented in `envs/drift_wrapper.py`.

#### Fixed food

Primary experimental mode:

- food levels sampled independently of agent composition
- default distribution: `{2: 0.6, 3: 0.4}`

This is the standard phase-one setting and is used in training configs.

#### Coupled food

Ablation mode for drift evaluation:

- food levels sampled around the mean agent level of the current episode
- `coupled_concentration = 0.7`

This was added to test whether the fixed-food drift protocol was confounded by changing effective task difficulty.

## 5. Drift Model

The non-stationary population process is an OU process over the type-frequency simplex. The implementation lives in `drift/ou_process.py` and is used through `envs/drift_wrapper.py`.

At every episode reset:

1. The OU process advances by one step: `x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * N(0, I)`
2. The result is projected onto the probability simplex via Euclidean projection.
3. Agent types for the episode are sampled i.i.d. from the current frequency vector.
4. Food levels are sampled according to the active food mode.

The two OU parameters — `sigma` (drift magnitude) and `theta` (mean-reversion strength) — form the axes of the drift evaluation grid. The grid uses 10 sigma values and 5 theta values, for 50 cells total.

## 6. Model Variants

### 6.1 Baseline GPL (Q1, Q3_rw)

Standard GPL-Q as described by Rahman et al. The architecture consists of:

- **Type inference LSTM** (`type_net_agent`): maps per-agent input `B_j` to a continuous type embedding.
- **Agent model** (GNN + MLP): predicts teammate actions from type embeddings.
- **Joint action-value network**: coordination-graph Q-factor over individual and pairwise terms.

Training uses Algorithm 5 from Rahman et al. with online updates, 16 parallel environments, and epsilon-greedy exploration decaying from 1.0 to 0.05 over 4.8M steps.

### 6.2 Auxiliary level-prediction head (Q3-inf-aux)

Adds a small MLP classifier on top of the type embedding that predicts teammate level class (1, 2, or 3) via cross-entropy loss. The auxiliary loss is weighted by `aux_weight` (default 0.1) and added to the standard GPL loss. The head shares the `type_net_agent` encoder with the main policy.

### 6.3 EMA population context (Q3-inf-ema)

Adds an exponential moving average tracker that maintains a running mean of type embeddings across episodes (`alpha = 0.1`, effective memory ≈ 10 episodes). The EMA vector (dimension 16) is concatenated to every agent's observation, increasing `obs_dim` from 11 to 27.

### 6.4 Combined (Q3-inf)

Uses both the auxiliary head and the EMA tracker. Effective `obs_dim = 27`.

## 7. Training Procedure

All training uses:

- 128,000 episodes (~6.4M environment steps at 50 steps/episode)
- 16 parallel environments with per-environment hidden state management
- Adam optimizer with learning rate 2.5e-4
- Gradient clipping at norm 10.0
- Polyak averaging for target network updates (tau = 1e-3)
- Epsilon decay from 1.0 to 0.05 over 4.8M steps

Training was performed on RTX 6000B GPUs via SLURM. Each training run produces a final checkpoint (`gpl_final.pt`) and periodic intermediate checkpoints.

## 8. Evaluation Procedure

### 8.1 Stationary greedy evaluation

- `sigma = 0.0`, `theta = 0.15`
- Greedy action selection (`epsilon = 0`)
- 500 episodes per seed
- Extended to 5 seeds (2500 episodes) for Q1 and Q3_rw sight sweep

### 8.2 Drift grid evaluation

- 10 sigma values × 5 theta values = 50 cells
- 100 episodes × 5 seeds = 500 episodes per cell
- 25,000 episodes per complete sweep
- Each cell records: mean return, IQM return, degradation relative to stationary baseline

### 8.3 Auxiliary diagnostics

- 12,000-episode short training runs for weight sweeps and sanity checks
- Raw cross-entropy logged per training step
- Bucketed into 3,000-episode windows for trend analysis

## 9. Metrics

- **Mean return**: arithmetic mean of episode returns.
- **IQM (Interquartile Mean)**: mean of the middle 50% of episode returns. More robust than mean to outlier episodes.
- **95% bootstrap CI for IQM**: 2,000 bootstrap resamples, percentile method.
- **Degradation**: fractional IQM loss relative to stationary baseline: `1 - IQM / IQM_baseline`.
- **Stability region**: fraction of (sigma, theta) grid points with degradation < 10%.
- **Cross-entropy (CE)**: raw level-prediction loss for auxiliary head diagnostics. Uniform three-class baseline: `ln 3 ≈ 1.099`.

## 10. Run Matrix

| Condition | Training config | Eval type | Seeds | Episodes |
|---|---|---|---:|---:|
| Q1 baseline | `gpl_lbf.yaml` | Stationary greedy | 5 | 2500 |
| Q2 fixed | Q1 checkpoint | Drift grid | 5 | 25000 |
| Q2 coupled | Q1 checkpoint | Drift grid (coupled food) | 5 | 25000 |
| Q3_rw sight=3 | `gpl_lbf_q3_rw.yaml` | Stationary greedy | 5 | 2500 |
| Q3_rw sight=4–7 | `gpl_lbf_q3_rw_sight{N}.yaml` | Stationary greedy | 5 | 2500 |
| Q3-inf-aux sight=3–7 | `gpl_lbf_q3_inf_aux*.yaml` | Stationary greedy | 1 | 500 |
| Q3-inf-ema | `gpl_lbf_q3_inf_ema.yaml` | Stationary greedy | 1 | 500 |
| Q3-inf | `gpl_lbf_q3_inf.yaml` | Stationary greedy | 1 | 500 |
| Q4_rw | Q3_rw checkpoint | Drift grid | 5 | 25000 |
| Q4-inf | Q3-inf checkpoint | Drift grid | 5 | 25000 |
| Q4-inf-aux | Q3-inf-aux checkpoint | Drift grid | 5 | 25000 |
| Q4-inf-ema | Q3-inf-ema checkpoint | Drift grid | 5 | 25000 |

## 11. Results Storage

All results are stored under `results/` (gitignored). Each evaluation produces:

- Per-episode CSV with return, episode length, agent levels, food levels, and OU state
- Summary CSV with per-cell aggregates (for drift grids)
- Heatmap plots (for drift grids)
- Baseline summary text file

Configs and SLURM scripts are version-controlled so that any run can be reproduced from the repository.

---

# Chapter 5: Results

This chapter reports the measured outcomes of the project and keeps them separate from interpretation. It follows the same broad sequence as the methodology chapter: stationary baselines, drift sweeps, sight sweeps, and auxiliary diagnostics. Explanations of why these patterns may have occurred are deferred to the analysis and discussion chapter.

## 1. Overview

The project produced four main groups of results:

1. **Stationary greedy evaluations** for the baseline (`Q1`), the information-nerfed baseline (`Q3_rw`), and the three inference variants (`Q3-inf-aux`, `Q3-inf-ema`, `Q3-inf`).
2. **Drift-sweep evaluations** for the baseline and nerfed families across a `10 x 5` Ornstein-Uhlenbeck grid, with one additional coupled-food re-run for `Q2`.
3. **Sight-sweep evaluations** for the RW baseline across sight radii `3-7`, plus a matching single-seed sight sweep for `Q3-inf-aux`.
4. **Auxiliary-head diagnostic runs** that measured raw level-prediction cross-entropy during shorter 12k-episode validation and ablation runs.

For the main stationary comparisons, the chapter reports mean return, interquartile mean (IQM), and a 95% bootstrap confidence interval for IQM. For the drift sweeps, the saved result artifact is one aggregate `(mean, IQM, degradation)` triple per grid cell; those cell values are reported directly below. Full raw artifacts remain in `results/`.

## 2. Stationary Baselines and Inference Variants

The first set of results comes from stationary greedy evaluation at `sigma = 0.0`, `theta = 0.15`, with 500 episodes per evaluation seed. `Q1` and the `Q3_rw` sight sweep were later extended to 5 seeds (2500 episodes total), while the three inference variants currently remain single-seed 500-episode evaluations.

| Family | Eval seeds | Episodes | Mean return | IQM return | 95% bootstrap CI for IQM |
|---|---:|---:|---:|---:|---:|
| `Q1` baseline | 5 | 2500 | 0.318 | 0.110 | [0.101, 0.136] |
| `Q3_rw` | 5 | 2500 | 0.347 | 0.160 | [0.150, 0.199] |
| `Q3-inf-aux` | 1 | 500 | 0.332 | 0.153 | [0.108, 0.180] |
| `Q3-inf-ema` | 1 | 500 | 0.304 | 0.115 | [0.093, 0.168] |
| `Q3-inf` | 1 | 500 | 0.292 | 0.115 | [0.097, 0.171] |

At the default stationary settings used for each family, the measured IQMs span `0.110-0.160`, and the measured mean returns span `0.292-0.347`.

## 3. Drift-Sweep Results

The drift evaluations were run over a `10 x 5` grid of `(sigma, theta)` settings. Each cell aggregates 5 evaluation seeds x 100 episodes, for 500 episodes per grid point and 25,000 episodes per sweep. The tables below summarize the stationary reference values at `sigma = 0`, the ranges observed across the full grid, and the number of grid cells whose degradation stayed below the 10% threshold used in the project.

### 3.1 Grid-wide summaries

| Family | Food mode | Grid cells | `sigma=0` mean | `sigma=0` IQM | Mean range across grid | IQM range across grid | Degradation range | Cells with degradation < 10% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Q2` | fixed | 50 | 0.351 | 0.152 | [0.350, 0.453] | [0.151, 0.356] | [-1.343, 0.004] | 50 / 50 |
| `Q2` | coupled | 50 | 0.418 | 0.230 | [0.410, 0.487] | [0.227, 0.417] | [-0.814, 0.013] | 50 / 50 |
| `Q4_rw` | fixed | 50 | 0.376 | 0.197 | [0.365, 0.481] | [0.188, 0.385] | [-0.957, 0.045] | 50 / 50 |
| `Q4-inf` | fixed | 50 | 0.287 | 0.135 | [0.287, 0.390] | [0.135, 0.271] | [-1.006, 0.000] | 50 / 50 |
| `Q4-inf-aux` | fixed | 50 | 0.360 | 0.189 | [0.359, 0.454] | [0.178, 0.357] | [-0.892, 0.054] | 50 / 50 |
| `Q4-inf-ema` | fixed | 50 | 0.317 | 0.141 | [0.303, 0.407] | [0.131, 0.259] | [-0.836, 0.069] | 50 / 50 |

### 3.2 Lowest- and highest-IQM grid cells

| Family | Lowest-IQM cell | Highest-IQM cell |
|---|---|---|
| `Q2` fixed | `(sigma=0.01, theta=0.30) -> mean 0.354, IQM 0.151, degradation 0.004` | `(sigma=1.00, theta=0.30) -> mean 0.450, IQM 0.356, degradation -1.343` |
| `Q2` coupled | `(sigma=0.01, theta=0.05) -> mean 0.410, IQM 0.227, degradation 0.013` | `(sigma=1.50, theta=1.00) -> mean 0.480, IQM 0.417, degradation -0.814` |
| `Q4_rw` fixed | `(sigma=0.01, theta=1.00) -> mean 0.365, IQM 0.188, degradation 0.045` | `(sigma=1.00, theta=0.50) -> mean 0.456, IQM 0.385, degradation -0.957` |
| `Q4-inf` fixed | `(sigma=0.00, theta=0.05) -> mean 0.287, IQM 0.135, degradation 0.000` | `(sigma=0.50, theta=0.05) -> mean 0.390, IQM 0.271, degradation -1.006` |
| `Q4-inf-aux` fixed | `(sigma=0.05, theta=0.30) -> mean 0.359, IQM 0.178, degradation 0.054` | `(sigma=0.50, theta=0.05) -> mean 0.454, IQM 0.357, degradation -0.892` |
| `Q4-inf-ema` fixed | `(sigma=0.01, theta=0.15) -> mean 0.303, IQM 0.131, degradation 0.069` | `(sigma=0.50, theta=0.50) -> mean 0.393, IQM 0.259, degradation -0.836` |

The complete per-cell values are stored in the corresponding `drift_eval_grid.csv` files under `results/`.

## 4. Sight-Sweep Results

The next set of results varies sight radius while keeping the stationary greedy evaluation protocol fixed. The `Q3_rw` sweep has multi-seed pooled evaluation (`5 x 500 = 2500` episodes per sight), while the currently available `Q3-inf-aux` sweep is single-seed (`1 x 500` episodes per sight).

### 4.1 RW sight sweep

| Family | Sight | Eval seeds | Episodes | Mean return | IQM return | 95% bootstrap CI for IQM |
|---|---:|---:|---:|---:|---:|---:|
| `Q3_rw` | 3 | 5 | 2500 | 0.347 | 0.160 | [0.150, 0.199] |
| `Q3_rw` | 4 | 5 | 2500 | 0.396 | 0.200 | [0.182, 0.271] |
| `Q3_rw` | 5 | 5 | 2500 | 0.335 | 0.147 | [0.137, 0.176] |
| `Q3_rw` | 6 | 5 | 2500 | 0.348 | 0.154 | [0.144, 0.181] |
| `Q3_rw` | 7 | 5 | 2500 | 0.330 | 0.145 | [0.131, 0.156] |

For reference, the full-observability `Q1` baseline measured mean return `0.318` and IQM `0.110` under the same stationary greedy evaluation protocol.

### 4.2 `Q3-inf-aux` sight sweep

| Family | Sight | Eval seeds | Episodes | Mean return | IQM return | 95% bootstrap CI for IQM |
|---|---:|---:|---:|---:|---:|---:|
| `Q3-inf-aux` | 3 | 1 | 500 | 0.332 | 0.153 | [0.108, 0.180] |
| `Q3-inf-aux` | 4 | 1 | 500 | 0.395 | 0.193 | [0.157, 0.283] |
| `Q3-inf-aux` | 5 | 1 | 500 | 0.229 | 0.118 | [0.062, 0.137] |
| `Q3-inf-aux` | 6 | 1 | 500 | 0.232 | 0.064 | [0.046, 0.099] |
| `Q3-inf-aux` | 7 | 1 | 500 | 0.364 | 0.159 | [0.131, 0.239] |

Across the measured sight settings, the highest stationary IQM in both sweeps was recorded at `sight = 4`.

## 5. Auxiliary-Head Diagnostic Results

The final result set comes from the shorter diagnostic runs used to inspect the auxiliary level-prediction head. These runs used raw level-prediction cross-entropy as the main metric. Lower values indicate better prediction. The information-free three-class baseline is `ln 3 ≈ 1.099`.

### 5.1 Weight-sweep bucket means (12k-episode runs, levels hidden)

| `aux_weight` | episodes 1-3k | episodes 3-6k | episodes 6-9k | episodes 9-12k |
|---:|---:|---:|---:|---:|
| 0.1 | 1.00 | 1.01 | 1.06 | 1.05 |
| 0.5 | 1.07 | 1.09 | 1.08 | 1.11 |
| 1.0 | 1.02 | 1.07 | 1.05 | 1.07 |
| 2.0 | 1.06 | 1.06 | - | - |

### 5.2 Summary of short-check runs

| Run | `aux_weight` | Mean CE | Median CE | Reduction vs uniform |
|---|---:|---:|---:|---:|
| levels-off (fixed auxiliary path) | 0.1 | 1.033 | 1.027 | -6% |
| levels-off (weight sweep) | 0.5 | 1.086 | 1.083 | -1% |
| levels-off (weight sweep) | 1.0 | 1.052 | 1.055 | -4% |
| levels-off (weight sweep) | 2.0 | 1.086 | 1.095 | -1% |
| levels-on (sanity check) | 0.1 | 0.902 | 0.866 | -18% |

These diagnostic results are reported separately from the main return-based evaluations because they measure classifier loss rather than task return, but they were part of the same empirical sequence and were used to document the behavior of the auxiliary module.

---

# Chapter 6: Analysis and Discussion

This chapter interprets the results reported in Chapter 5. The goal is to explain what the data mean, which Specific Aims were met, which hypotheses were supported or rejected, and what assumptions limit the strength of the conclusions. The main outcome of the project is not the one originally expected. The original proposal centered on population composition drift, but the strongest completed result is instead about observability: in this LBF setting, giving GPL more teammate information was not monotonically better, and a moderately restricted observation setting produced the best stationary performance.

That pivot is not a failure of the project. The Specific Aims were designed to characterize GPL under deployment stresses, and a characterization is valuable whether the measured behavior agrees with the initial hypotheses or contradicts them. The analysis below therefore separates three things:

1. what the experiments show directly;
2. what those results imply about GPL, observability, and drift in the current setup;
3. what cannot yet be claimed without additional experiments.

## 1. Context from Previous Work

Open ad hoc teamwork (AHT) asks a learning agent to cooperate with teammates whose types, capabilities, or policies may be unknown at test time. GPL is a natural starting point for this project because it was designed for open AHT: it represents teammate information with recurrent type embeddings, predicts teammate actions with a graph-based agent model, and factors the joint action-value function over a coordination graph.[^rahman] ODITS and related work also treat teammate modeling under partial observability as a central problem.[^gu] This literature establishes that latent teammate inference is important, but most evaluations still assume that the distribution producing teammates is stationary.

The original motivation for this project came from that gap. If teammates are drawn from a distribution that changes over time, then a stationary-trained AHT policy might degrade as the test distribution moves away from its training support. Work on non-stationary teammates, such as Santos et al. and Zhang et al., studies changing teammate behavior or sudden policy shifts, but not the particular case of a continuously drifting population composition distribution.[^santos][^zhang] The OU drift wrapper implemented here was meant to make that missing axis measurable.

The completed phase-one experiments show that the drift axis is not diagnostic in the present LBF setup, but they reveal a different and useful gap in the literature. Partial observability is usually treated as an obstacle: less information should make teammate inference harder. In contrast, the stationary sight sweep shows that more information is not automatically better for this GPL setup. This places the result closer to a broader machine-learning idea: limiting information can sometimes improve generalization by suppressing nuisance variables, acting like a bottleneck or regularizer. The contribution is not that partial observability is always good, but that observability should be treated as a design variable rather than only as an environmental handicap.

## 2. Specific Aims

### 2.1 Aim 1: Establish stationary baselines

Aim 1 was met. The project trained and evaluated the baseline GPL configuration (`Q1`) and the information-nerfed GPL configuration (`Q3_rw`) under stationary greedy evaluation. This gave a reference point for later drift and observability comparisons.

The important result is that the information-nerfed setting did not simply underperform the full-observability baseline. In the multi-seed stationary evaluation, `Q1` achieved mean return `0.318` and IQM `0.110`, while the default `Q3_rw` setting at `sight = 3` achieved mean return `0.347` and IQM `0.160`. The wider `Q3_rw` sight sweep showed its best result at `sight = 4`, with mean return `0.396` and IQM `0.200`.

This outcome changes the interpretation of the baseline comparison. The project began from the assumption that hiding teammate levels and reducing sight would make the task harder. That assumption is partly right in an information-theoretic sense: hidden levels are harder to infer. But the control result shows that the policy can perform better under the restricted observation regime than under the full-observability baseline. The stationary baseline comparison therefore supports a more subtle claim: the useful amount of teammate information for this GPL-LBF setup is not simply "as much as possible."

### 2.2 Aim 2: Characterize drift stability regions

Aim 2 was also completed procedurally. The drift wrapper, OU process, 50-cell grid, degradation metric, and heatmap artifacts were implemented and run for the baseline and information-nerfed families. The result, however, does not support the original drift-degradation hypothesis.

For `Q2`, `Q4_rw`, and the inference variants, all 50 grid cells stayed below the 10 percent degradation threshold. Several drift cells had negative degradation, meaning their IQM was higher than the stationary reference for that sweep. The coupled-food re-run of `Q2` was added to test whether fixed food was producing a capability confound, but it also did not produce a meaningful degradation boundary.

The conclusion is therefore conservative: in the present 8x8 LBF setup, these trained GPL checkpoints do not show a clear drift-induced collapse across the tested OU grid. This does not prove that GPL is generally robust to population composition drift. It only says that this particular protocol did not make drift a binding stressor. The drift infrastructure remains a useful engineering product of the project, but the drift result is a non-result for the current scientific claim.

### 2.3 Aim 3: Characterize observability through the sight sweep

Aim 3 became the central empirical aim. The original version of the aim predicted a threshold: as sight increased, the auxiliary level-prediction head should eventually gain enough behavioral evidence to infer hidden teammate levels, and stationary return should improve. The measured results do not support that monotone threshold story.

The `Q3_rw` sight sweep is the cleanest evidence because it has 5 evaluation seeds and 2500 episodes per sight level. IQM rises from `0.160` at `sight = 3` to `0.200` at `sight = 4`, then falls to `0.147`, `0.154`, and `0.145` at sights `5`, `6`, and `7`. The single-seed `Q3-inf-aux` sight sweep is noisier, but its best measured IQM is also at `sight = 4`.

This result falsifies the strongest form of the "more sight is better" hypothesis. The most important part is not just that `sight = 4` is best among the measured points. The more important fact is the shape of the curve: relaxing the observation from `sight = 3` helps, but continuing to relax it does not keep helping. In this phase-one setup, observability behaves like a non-monotone design variable.

## 3. Hypotheses

### 3.1 H1: Drift degradation

H1 predicted that, holding observability fixed, a stationary-trained GPL policy would degrade as drift magnitude increased and mean reversion weakened. The measured drift grids do not support this hypothesis in the current setup.

The strongest evidence against H1 is that every measured drift sweep remained inside the predefined stability region. The degradation metric never crossed the 10 percent threshold for `Q2`, `Q4_rw`, or the inference variants. The coupled-food `Q2` re-run also remained stable by the same criterion. Because the same trained checkpoints were used across the grid, the result says that evaluation-time OU composition drift did not reliably harm return under this task protocol.

This should be interpreted as a failure of the tested environment-protocol pair to expose drift fragility, not as a broad theorem about GPL. LBF rewards depend on the interaction between agent levels, food levels, visibility, and random teammate behavior. In the fixed-food protocol, changing the population can also change effective task difficulty. In the coupled-food protocol, the food distribution is adjusted toward current team capability, which removes one confound but also makes the evaluation less adversarial. Neither variant produced the degradation surface H1 predicted.

### 3.2 H2: Observability threshold for teammate inference

H2 predicted an observability threshold below which teammate-level inference would be infeasible and above which the auxiliary head would begin to extract level information and improve return. The results partially support the first half but reject the full threshold hypothesis.

The diagnostic runs support the idea that teammate level is hard to infer when levels are hidden. With levels hidden, the auxiliary head's raw cross-entropy stayed close to the uniform three-class baseline of `ln 3 = 1.099`, even when `aux_weight` was increased from `0.1` to `2.0`. The levels-on sanity check reduced mean CE to `0.902`, showing that the code path can use level information when it is directly available. That supports the claim that the levels-off condition contains limited usable signal for the auxiliary classifier.

The sight sweep, however, rejects the simple threshold story. If the threshold hypothesis were correct in its strongest form, we would expect increasing sight to produce either a clear monotone improvement or a visible transition after which performance remains better. Instead, performance improves at moderate sight and then declines or flattens. The better interpretation is that there are two constraints, not one: too little information prevents useful inference, but too much visible structure can introduce nuisance features that do not help the policy's actual objective.

### 3.3 H3: Drift and observability interaction

H3 predicted that drift would be more damaging under restricted observability than under full observability. The current evidence does not support this hypothesis.

The reason is simple: because the drift sweeps did not produce meaningful degradation in either setting, there is no reliable shrinkage of the stability region to compare. `Q4_rw` did not reveal a smaller stability region than `Q2` under the tested grid. This means the project cannot claim an empirical interaction between OU population drift and observability.

The hypothesis remains scientifically plausible, but it requires a setup in which drift matters in the first place. A harder environment, more capable non-random teammates, a different food-generation protocol, or a drift process that changes behaviorally meaningful teammate mixtures may still expose the interaction. In the present work, H3 is best treated as an unconfirmed future-work direction rather than a supported result.

## 4. Why the Inference Remedies Did Not Become the Main Result

The project tested three inference-oriented variants: an auxiliary level-prediction head, an EMA population context, and the combination of both. None clearly improved on the `Q3_rw` baseline.

The auxiliary head was motivated by a reasonable idea from AHT: if teammate types are hidden, an auxiliary supervised signal might force GPL's recurrent type embeddings to encode capability. The first implementation issue was that the auxiliary forward pass initially reset the recurrent hidden state, so the classifier saw only a single-frame embedding. Under `observe_agent_levels = false`, a single frame contains position but not level, making near-uniform prediction expected. Fixing the hidden-state path allowed the head to vary, but the mean CE stayed only slightly below uniform in levels-off runs.

The deeper issue is multi-task interference. The auxiliary head reads from the same `type_net_agent` representation that the GPL agent model uses for teammate action prediction. In these runs, teammates are random. Their action distributions are not strongly level-dependent, so the dominant action-prediction objective has little reason to preserve level information. A level-1 random teammate and a level-3 random teammate may differ in what they could do, but if their behavior policy is random, their observed action stream does not reliably advertise that capability. The auxiliary loss therefore asks the shared encoder to preserve information that the main modeling loss does not value.

The EMA tracker inherits the same problem. It averages learned type embeddings across episodes and passes that context back to the policy. If the embeddings do not reliably encode teammate capability, the EMA context cannot become a useful population summary. It also increases the policy input dimension, so it can add noise or distribution shift without adding a stable signal. This explains why `Q3-inf-ema` and `Q3-inf` did not outperform the plain `Q3_rw` baseline.

The failed remedies are still informative. They show that adding an auxiliary label or a cross-episode memory module is not enough by itself. The representation being supervised and averaged must actually contain behaviorally relevant information, and the main policy objective must not be indifferent to the feature the auxiliary module is trying to recover.

## 5. Why Moderate Observability Can Help

The central interpretive claim of the project is that observability in open AHT can act as an information bottleneck. The strongest measured fact is the non-monotone `Q3_rw` sight curve: `sight = 4` is better than `sight = 3`, but larger sight values do not keep improving performance.

A useful way to understand this is to separate signal from availability. At very low sight, the policy may not observe enough teammate movement or food context to coordinate well. Increasing from `sight = 3` to `sight = 4` likely increases the amount of useful local information available during each 50-step episode. But additional visibility also increases the amount of state the model must process. Some of that extra state may be weakly related or unrelated to the action-value decision. If GPL's learned representation attends to features that are easy to model but not useful for control, then more observation can hurt rather than help.

This connects to the broader idea of information throttling. In supervised and reinforcement learning, bottlenecks, dropout, attention masks, and state abstraction can improve generalization by preventing a model from relying on high-variance or nuisance features. The result here is an AHT-specific version of that idea: the teammate observation channel is not only a source of useful evidence, but also a source of distractors. Under this task distribution, moderate restriction appears to preserve enough local coordination information while removing some unhelpful structure.

This does not mean that partial observability is generally better. It means that "more observability" and "more useful information" are not the same. A representation-learning algorithm can be harmed by additional variables when those variables are weakly aligned with the training objective, particularly in a small environment where random teammate behavior limits the connection between capability and observed action.

## 6. Assumptions Behind the Interpretation

Several assumptions shape the analysis and should be made explicit.

First, the project assumes IQM is the most appropriate primary metric for these returns. Mean return is also reported, but LBF returns have many low-return or zero-return episodes, so IQM gives a more robust view of typical performance. This choice matters because some comparisons look different in mean return than in IQM. The analysis therefore treats IQM as the main metric and mean return as supporting context.

Second, the strongest stationary comparisons use evaluation seeds, not independent training seeds. `Q1` and the `Q3_rw` sight sweep were evaluated over 5 seeds and 2500 episodes per setting, which reduces rollout noise, but each point still comes from a trained checkpoint or checkpoint family rather than a full multi-training-seed study. This is acceptable for phase-one characterization but weaker than a publication-level estimate of training variability.

Third, the comparison between `Q1` and `Q3_rw` changes two observability properties at once: sight radius and direct access to teammate levels. The configs keep the environment structure aligned on grid size, agent count, food count, horizon, and forced cooperation, but `Q1` has `sight = 8` and `observe_agent_levels = true`, while `Q3_rw` has reduced sight and `observe_agent_levels = false`. The `Q3_rw` sight sweep isolates sight radius only within the levels-hidden regime. It does not by itself disentangle the effect of hiding levels from the effect of changing sight.

Fourth, the teammates used in this setup are random. That assumption is important because level is a capability variable, but random behavior may not express capability in a way that the action-prediction objective can learn. If teammates used level-dependent policies, the same auxiliary head and GPL type encoder might receive a stronger behavioral signal.

Fifth, the drift evaluation changes population composition at episode boundaries while keeping the trained checkpoint fixed. This is exactly what the project intended to test, but it means drift results should be interpreted as evaluation-time robustness, not as online adaptation during training. The EMA module is a lightweight context mechanism, not a full drift-aware retraining or Bayesian filtering method.

Sixth, the coupled-food ablation changes the meaning of the drift stressor. It was useful for testing whether fixed food created a capability confound, but it also makes task difficulty follow the current team. That is a different evaluation question from asking whether a stationary policy can survive adverse composition changes under a fixed task distribution.

## 7. Limitations

The main limitation is scope. The completed experiments are all in Level-Based Foraging. The original proposal mentioned broader MARL evaluation and additional environments, but the phase-one evidence comes from one environment family, one grid size, one agent count, and one main GPL-Q implementation. A general claim about open AHT should be replicated in at least one additional task.

The second limitation is seed coverage. The `Q3_rw` sight sweep and `Q1` evaluation have stronger multi-seed rollout evidence, but the inference variants and the `Q3-inf-aux` sight sweep remain single-seed evaluations. Their patterns are useful for diagnosis, but they should not be treated as final effect-size estimates.

The third limitation is that the sight sweep is not yet dense enough. The measured sights `3-7` show a non-monotone shape, but they do not determine whether the optimum is specifically `sight = 4`, a broad low-sight plateau, or part of a more complex curve. A denser sweep including `sight = 2` and `sight = 8` under the same levels-hidden condition would better locate the response surface.

The fourth limitation is that the auxiliary diagnostics measure level-prediction CE, not direct mutual information or causal usefulness for control. A low CE would show that level is recoverable, but a high CE does not prove that no useful teammate information is present. It only shows that this classifier, using this shared encoder, did not extract the level labels reliably.

The fifth limitation is the random-teammate protocol. Random teammates are useful for reproducing the GPL-style setup and for isolating structural effects, but they are a weak test of teammate capability inference. Many real AHT settings involve teammates whose actions are strongly tied to skill, intent, role, or policy class. The conclusions about auxiliary inference may change under those richer teammate distributions.

The sixth limitation is that drift did not become binding. Since the drift grid did not produce degradation, the project cannot evaluate recovery, collapse boundaries, or drift-aware adaptation in a meaningful way. The OU process and drift wrapper are valid infrastructure, but the current task protocol is not hard enough to support strong claims about robustness to population composition drift.

## 8. Implications for the Final Conclusion

The final conclusion should be framed as a negative result with a constructive design lesson.

The negative part is clear: the original drift-degradation hypothesis was not supported in the present setup, the drift-observability interaction was not observed, and the proposed auxiliary/EMA remedies did not outperform the plain `Q3_rw` baseline. These are real outcomes and should not be hidden.

The constructive part is stronger: the project found that observation design materially affects GPL performance, and that the relationship is non-monotone in the measured LBF setting. This is a useful contribution because it challenges the default assumption that partial observability is only a deficit. For GPL-style open AHT, the observation channel can include nuisance information, and limiting that channel can improve the learned policy.

The project therefore meets its aims by characterizing GPL under the planned stresses, even though the strongest result came from falsifying the original expectations. The final thesis should argue that phase one establishes observability as the more promising axis for future work: learned masking, richer teammate policies, denser sight sweeps, and multi-environment replication are the natural next steps. Drift remains important as a research direction, but in this project it is best presented as implemented infrastructure and a non-diagnostic result under the current LBF protocol, not as the central empirical claim.

---

[^rahman]: Arrasy Rahman et al., "A General Learning Framework for Open Ad Hoc Teamwork Using Graph-Based Policy Learning," *Journal of Machine Learning Research* 24, no. 298 (2023): 1-74.

[^gu]: Peihao Gu et al., "Online Ad Hoc Teamwork under Partial Observability," in *Proceedings of the International Conference on Learning Representations (ICLR)* (2022).

[^santos]: Pedro M. Santos et al., "Ad Hoc Teamwork in the Presence of Non-Stationary Teammates," in *Progress in Artificial Intelligence: EPIA 2021*, Springer LNCS 12981 (2021), 648-60.

[^zhang]: Ziqian Zhang et al., "Fast Teammate Adaptation in the Presence of Sudden Policy Change," in *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 216 (2023), 2465-76.

---

# Chapter 7: Conclusion

This project investigated how Graph-based Policy Learning (GPL) behaves when open ad hoc teamwork is evaluated under two deployment stresses: restricted observability of teammates and non-stationary population composition. The original motivation was that real multi-agent systems do not always encounter teammates drawn from a fixed distribution. Human teams change, robot fleets become heterogeneous, and deployed agents may face shifting mixtures of teammate capabilities. The project therefore set out to build an evaluation framework for population composition drift and to test whether GPL would degrade under that stress, especially when teammate information was partially hidden.

The main conclusion is that the strongest finding of the project is not about drift collapse, but about observation design. In the Level-Based Foraging setup studied here, increasing teammate observability did not produce a simple monotone improvement. Instead, the best stationary behavior appeared under a moderately restricted observation setting. This means that, for this GPL setup, more teammate information was not automatically more useful information. Some information may help coordination, while additional visible structure may act as a distraction or nuisance variable for the learned representation.

This finding changes the way partial observability should be understood in this project. At the start, partial observability was treated mainly as a handicap: hiding teammate levels and reducing sight should make inference harder and therefore reduce performance. The results support only part of that assumption. Hidden teammate levels were indeed difficult for the auxiliary classifier to recover, especially when teammate behavior was random. However, the control policy did not simply improve as observability increased. The more useful conclusion is that observability is a design choice. A policy's input should not be judged only by how much state it contains, but by whether the included state is aligned with the task and the learning objective.

The drift experiments also produced an important conclusion, although it is a negative one. The project successfully implemented an Ornstein-Uhlenbeck drift process over teammate type frequencies, connected it to the LBF environment through a drift wrapper, and evaluated trained GPL checkpoints across a systematic drift grid. Under the present protocol, those sweeps did not reveal a meaningful degradation boundary. A coupled-food ablation was also tested to address the possibility that fixed food difficulty was confounding the results, but it likewise did not make drift a binding stressor. Therefore, this project should not claim that GPL is generally robust to population composition drift. The stronger and more accurate conclusion is that the current LBF setup is not sensitive enough to support a drift-robustness claim.

The auxiliary inference mechanisms were also informative, even though they did not produce the hoped-for improvement. The auxiliary level-prediction head and the EMA population context were designed to help GPL recover hidden teammate structure. In practice, they did not outperform the plain information-nerfed baseline. The diagnostic evidence suggests that this is not just a tuning issue. The auxiliary head shares an encoder with GPL's teammate action-prediction objective, and random teammate behavior gives that objective little reason to preserve level information. The EMA tracker then averages embeddings that do not reliably encode the population property it is supposed to summarize. The result is a useful design lesson: adding supervision or memory is not enough if the representation being supervised does not contain the right signal.

Taken together, the project met its Specific Aims by building the experimental infrastructure, reproducing stationary GPL baselines, evaluating drift stability, and characterizing observability through sight sweeps and auxiliary diagnostics. The hypotheses were not all supported, but the experiments answered the questions they were designed to ask. The drift-degradation hypothesis was not supported in this environment. The monotone observability-threshold hypothesis was also rejected in its strongest form. The drift-by-observability interaction could not be established because drift itself did not create a meaningful degradation region. These outcomes are still valuable because they narrow the claims that can be made and point to a clearer research direction.

The most important contribution of this phase is therefore a reframing: in open ad hoc teamwork, observation design should be treated as a first-class experimental and algorithmic variable. The project shows that restricted observability can sometimes improve performance, not because ignorance is generally beneficial, but because the learning system may benefit from a better-shaped information bottleneck. This connects the work to a broader principle in machine learning: useful representations often depend as much on excluding nuisance information as on including relevant information.

There are several limitations. The experiments were performed in one environment family, with one main GPL-Q implementation and a limited set of teammate behaviors. The strongest stationary comparisons use multiple evaluation seeds, but not a full multi-training-seed study. The inference-variant sight sweep is still less statistically complete than the RW sight sweep. The drift result should also be viewed as protocol-specific, because a harder environment, more capable teammates, or a different task-generation process may produce different behavior. These limitations do not undermine the main conclusion, but they define its scope.

Future work should build directly on the observability result. The next step is not simply to run larger versions of the same drift sweep, but to understand which parts of the observation are useful and which parts are harmful. A learned observation mask, a denser sight sweep, richer non-random teammate policies, and replication in another cooperative environment would all test whether the observed non-monotonicity is a general property of GPL-style open AHT or a feature of this particular LBF setup. Drift should remain part of the long-term agenda, but it should be studied in a setting where changing population composition actually changes the difficulty faced by the policy.

In summary, the project began as a stress test for composition drift and ended with a sharper lesson about information. GPL did not fail under the tested drift protocol, but it did reveal that the amount and form of teammate information strongly shape performance. The final takeaway is that robust open ad hoc teamwork requires more than powerful teammate-inference machinery. It also requires careful control over what the learner is allowed to observe, because the observation interface itself can determine whether the learned representation supports cooperation or is pulled toward irrelevant structure.

---

# Chapter 8: Recommendations

This chapter describes the work that follows directly from the thesis conclusions and is specifically intended to guide the continuation of this research into a publication. Each recommendation addresses a gap or limitation identified in the analysis, is scoped to a concrete experiment or engineering task, and is prioritized by how much it would strengthen the paper's central claim: that observation design is a non-monotone, first-class variable in open ad hoc teamwork.

The recommendations are organized into three tiers. Tier 1 items are essential for a convincing publication and should be completed first. Tier 2 items would meaningfully strengthen the paper's claims or broaden its scope. Tier 3 items are valuable extensions that are not strictly necessary for publication but would open new research threads.

---

## Tier 1: Essential for Publication

### 1.1 Learned observation masking

The thesis showed that hand-tuned sight radius produces a non-monotone performance curve, with moderate restriction outperforming both the most restricted and the most permissive settings. The natural next step is to ask whether a learned module can recover this sweet spot automatically. This is the single most important recommendation because it converts the thesis's analytical finding into an algorithmic contribution.

The proposed mechanism is a differentiable observation mask or gating module placed between the raw LBF observation and GPL's PREPROCESS input. The mask would be trained end-to-end with the policy, learning which spatial and feature dimensions to pass through. If the learned mask converges to a moderate-information configuration similar to the hand-set optimum, it confirms that the benefit of restriction is about suppressing nuisance features, not about a quirk of the sight parameter. If the mask converges to a different configuration, that itself is informative: it reveals which specific observation components are helpful or harmful.

This recommendation addresses the thesis limitation that the sight sweep is a coarse, one-dimensional proxy for information content. A learned mask searches over a richer space of observation structures.

**Engineering scope.** The existing `envs/env_utils.py` PREPROCESS pipeline already separates agent features from food features, so the mask can operate on the pre-decomposed input vectors. The mask module is a small neural network (a few linear layers with sigmoid output) whose parameters are updated alongside the policy. No changes to the environment or the training loop's outer structure are needed; only the observation path and the loss computation require modification.

### 1.2 Multi-environment replication

The thesis evidence comes entirely from Level-Based Foraging. A reviewer's most natural objection is that the non-monotone observability curve might be an artifact of LBF's specific reward structure, grid geometry, or random-teammate protocol. Replicating the sight sweep in at least one additional cooperative partial-observation environment would establish that the finding is a property of GPL-style open AHT rather than of this single benchmark.

The two most practical candidates are:

- **Overcooked.** A cooperative cooking task with spatial coordination, partial observability, and teammate-dependent reward. It has an existing GPL-compatible observation format and is well-established in the AHT literature.
- **MPE cooperative navigation.** A continuous-space cooperative task where agents must cover landmarks. Partial observability can be introduced by limiting communication range or sensor radius, giving a natural analog to the sight-radius manipulation in LBF.

In either case, the experiment is a sight sweep (or sensor-range sweep) under the same GPL-Q training and stationary greedy evaluation protocol used in the thesis. If a non-monotone curve appears in a second environment, the publication can claim generality. If it does not, the paper should explain which structural differences account for the discrepancy.

**Engineering scope.** The main cost is implementing PREPROCESS for the new environment and writing a training config. The GPL architecture, training loop, evaluation code, and drift wrapper are environment-agnostic by design. A single training run plus a five-point sight sweep with multi-seed evaluation is comparable in compute to one quadrant of the thesis experiments.

### 1.3 Denser sight sweep with multi-seed training

The thesis sight sweep covers five points (`sight = 3, 4, 5, 6, 7`) with multi-seed evaluation but single-seed training. Two extensions are needed for publication strength:

First, add `sight = 2` and `sight = 8` under the levels-hidden condition. `sight = 2` tests whether performance continues to degrade below the current minimum, and `sight = 8` closes the gap between the RW regime and the Q1 baseline by using the same grid-wide visibility but without direct level access. Together, these two points complete the response surface from maximally restricted to maximally permissive within the levels-hidden regime.

Second, repeat training with at least three independent training seeds at each sight level. The current results use multiple evaluation seeds to reduce rollout noise, but they do not account for training variability. Multi-training-seed results are necessary for reporting confidence intervals that reflect the full experimental pipeline, not just the evaluation stage.

**Engineering scope.** Each training run is one 128k-episode GPU job. With seven sight levels and three training seeds, this is twenty-one training runs plus corresponding evaluations. The existing parameterized SLURM scripts (`q3_rw_multiseed_greedy_eval.slurm`, `submit_rw_multiseed_eval.sh`) can be extended to handle this with minimal modification.

---

## Tier 2: Strengthens the Paper

### 2.1 Non-random teammate policies

The thesis used random teammates throughout, following the standard GPL training protocol. Random behavior is useful for isolating structural effects, but it also limits what the auxiliary head and the policy can learn about teammate capability. A level-1 random teammate and a level-3 random teammate produce similar action distributions, which is why the action-prediction objective is level-indifferent and the auxiliary head cannot extract level information.

Training and evaluating with a fraction of non-random teammates — for example, teammates that follow simple heuristic policies correlated with their level — would test whether the auxiliary head becomes useful when level is behaviorally expressed. This recommendation directly addresses the multi-task-interference mechanism identified in the thesis. If the auxiliary head learns under non-random teammates, it confirms that the thesis failure is about the training distribution, not about the architecture. If it still fails, the architectural limitation is deeper than the thesis analysis suggests.

**Engineering scope.** The teammate behavior is controlled by the evaluation and training loops, which sample actions for non-learner agents. Replacing the random policy with a simple level-dependent heuristic (e.g., higher-level agents prefer food they can collect alone) requires a small addition to the training script, not a change to the GPL agent or environment.

### 2.2 Separate encoder for the auxiliary head

The thesis identified multi-task interference as a key reason the auxiliary level-prediction head failed: it shares `type_net_agent` with the dominant action-prediction objective, and the two gradients pull the shared representation in different directions. A clean test of this mechanism is to give the auxiliary head its own dedicated encoder — a second small LSTM that reads the same observation stream but whose parameters are not updated by the action-prediction loss.

If the separate-encoder auxiliary head learns to predict level while the shared-encoder version does not, the interference hypothesis is confirmed and the paper gains a concrete architectural recommendation: auxiliary supervision in open AHT should use a decoupled representation. If it still fails, the information floor is the binding constraint regardless of architecture.

**Engineering scope.** This is a moderate code change in `agents/gpl/gpl_agent_inf.py`. A second `type_net_agent`-like LSTM is instantiated, its output is fed to the existing auxiliary head, and its parameters are included in the optimizer. The rest of the training loop remains unchanged. One training run at the default sight level is sufficient for the diagnostic comparison.

### 2.3 Information-theoretic bound on teammate-level inference

The thesis provides empirical evidence that teammate level is hard to infer from behavioral observation under low sight and random teammates, but it does not give a theoretical bound. Even a loose mutual-information argument — bounding `I(trajectory; teammate_level)` as a function of sight radius, episode length, and teammate-policy entropy — would anchor the empirical curve in theory and distinguish between "hard to learn" and "informationally impossible."

This would elevate the paper from a descriptive empirical study to one with a mechanistic explanation. The bound does not need to be tight; it needs to track the measured cross-entropy qualitatively and to predict the approximate location of the transition from information-limited to representation-limited regimes.

**Engineering scope.** This is primarily an analytical task, not an engineering one. The key inputs are the LBF observation structure, the visibility geometry at each sight level, and the action-space entropy of random teammates. A back-of-envelope calculation followed by a Monte Carlo verification against the training data would be sufficient.

### 2.4 `aux_weight` sweep on the levels-on side

The thesis ran the `aux_weight` sweep only in the levels-off condition, where the information floor dominates and weight has no effect. Running the same sweep with `observe_agent_levels = true` would isolate the multi-task interference term: if CE falls monotonically with increasing weight when level is directly observable, the interference is the binding constraint and can in principle be overcome by rebalancing the loss. If CE plateaus even at high weight, the shared-encoder architecture has a deeper capacity issue.

This is a small experiment that produces a clean figure for the paper and directly complements the levels-on sanity check already in the thesis.

**Engineering scope.** Four to five short 12k-episode training runs at different `aux_weight` values, using the existing `configs/gpl_lbf_q3_inf_aux_shortcheck_levels_on.yaml` config as a template. The aggregation and plotting code from the thesis diagnostic series can be reused directly.

---

## Tier 3: Valuable Extensions

### 3.1 Input augmentation with teammate action history

The thesis observation under the information nerf includes only teammate position at the current timestep. Including the previous action of each visible teammate as an additional input feature would restore a channel that correlates with capability: a teammate that consistently uses the "load" action near food is behaviorally different from one that moves randomly. This is a principled counterfactual for the information-floor hypothesis. If the auxiliary head learns under action-augmented observations even at low sight, the bottleneck was the observation content, not the method.

**Engineering scope.** A refactor of `envs/env_utils.py` to append previous teammate actions to the agent feature vector, plus retraining. The observation dimension increases, and the PREPROCESS decomposition must be updated, but the GPL architecture handles variable input dimensions by design.

### 3.2 Alternative inference architectures

The thesis tested one auxiliary supervision approach (cross-entropy level prediction) and one cross-episode memory approach (EMA). Other architectures may perform differently:

- **Attention-based context module.** Replace the EMA mean with a learned attention aggregation over recent teammate type embeddings, allowing the model to weight informative episodes more heavily.
- **Explicit Bayesian type posterior.** Maintain a per-teammate posterior over level classes, updated online within and across episodes using observed behavior as likelihood. This separates the inference problem from the representation-learning problem.
- **Contrastive teammate embedding.** Train the type encoder with a contrastive loss that pulls same-level teammates together and pushes different-level teammates apart, rather than predicting level labels directly.

Each alternative tests whether the thesis negative result is specific to the auxiliary-head-plus-EMA family or is fundamental to the information regime. Either outcome strengthens the paper.

**Engineering scope.** Each alternative is a self-contained module that plugs into the existing `GPLAgentInf` class. The attention and contrastive variants are small additions; the Bayesian posterior requires more careful design but no changes to the environment or evaluation infrastructure.

### 3.3 Drift in a harder environment

The thesis drift infrastructure — OU process, drift wrapper, grid sweep, degradation metric, heatmaps — is fully implemented and tested, but the current LBF setup did not make drift a binding stressor. If the publication includes a second environment (Recommendation 1.2), the drift grid should be re-run in that environment to test whether composition drift becomes meaningful under a harder task.

The thesis conclusion explicitly notes that the LBF protocol may be too easy for drift to matter: fixed food levels create a capability confound, and coupled food makes the task track the team. A task where food difficulty is fixed and high, or where coordination is mandatory rather than optional, would more likely expose a degradation boundary.

**Engineering scope.** The drift wrapper is environment-agnostic. Once PREPROCESS is implemented for the new environment, the drift evaluation code can be run without modification. The compute cost is one 50-cell grid sweep per checkpoint.

### 3.4 Population-conditioned policy

The EMA tracker failed because it averaged embeddings that did not encode the population property it was meant to summarize. A stronger version would replace the fixed EMA with a learned nonlinear population encoder trained end-to-end. However, this recommendation is conditional on first establishing that the type embeddings contain usable population signal — either through non-random teammates (Recommendation 2.1), a separate encoder (Recommendation 2.2), or a harder environment (Recommendation 3.3). Without that prerequisite, a more powerful aggregator will still be averaging uninformative representations.

### 3.5 Explicit change-point detection for drift adaptation

Methods such as CUSUM on type-embedding statistics, combined with policy switching or reweighting, are a natural approach to drift-aware adaptation. Like the population-conditioned policy, this recommendation is deferred until the embeddings carry signal worth monitoring. The drift infrastructure from the thesis is ready to support these experiments once the prerequisite is met.

---

## Priority Roadmap for the Publication

The following sequence represents the recommended order of work for turning the thesis into a publication:

1. **Denser sight sweep with multi-seed training** (1.3) — strengthens the existing central result with minimal new engineering.
2. **Multi-environment replication** (1.2) — addresses the single most likely reviewer objection.
3. **Learned observation masking** (1.1) — converts the analytical finding into an algorithmic contribution, which is the publication's distinguishing feature.
4. **Non-random teammates** (2.1) and **separate auxiliary encoder** (2.2) — together, these two experiments resolve the multi-task-interference mechanism and determine whether the auxiliary approach has a future in this line of work.
5. **Information-theoretic bound** (2.3) — adds theoretical grounding that elevates the paper above a purely empirical study.
6. **Remaining Tier 3 items** as time and compute allow.

This ordering front-loads the experiments that reduce risk (stronger existing evidence, second environment) before investing in the algorithmic contribution (learned masking) that is the publication's main novelty. The Tier 2 mechanistic experiments then provide the explanatory depth that reviewers expect, and the Tier 3 extensions can be included or deferred depending on the submission timeline.
