# Literature Review: From Ad Hoc Teamwork to Drift and Partial Observability

This review orients the reader toward the MQP’s actual design: **GPL** under **population composition drift**, with a controlled **information nerf** that makes teammate capability only partially observable. The narrative keeps two difficulties—**limited sensing of teammates** and **non-stationarity of the composition-generating process**—on separate axes so neither is confused for the other.

---

## 1. Ad Hoc Teamwork as the Common Problem

Stone and colleagues formalized *ad hoc autonomous agent teams*: a single learner must collaborate with previously unknown teammates, without pre-coordination, on tasks where every agent can contribute.[^stone] Subsequent work expanded AHT to open settings (variable team size, joining and leaving, unknown teammate policies) and surveyed a large body of methods and benchmarks.[^mirsky] That survey usefully stresses evaluation diversity; it does not, by itself, prescribe how policies should behave when the **distribution** from which teammates are drawn **changes over time**—the gap this project stresses.

---

## 2. Two Axes of Difficulty (Kept Distinct)

Many MARL discussions collapse “hard multi-agent settings” into one bucket. For this project, two structurally different axes matter.

| Axis | What it refers to | How the literature usually frames it | How this MQP instantiates it |
|------|-------------------|--------------------------------------|-------------------------------|
| **A. Partial observability of teammate structure** | The focal agent’s observations do not fully identify teammates’ capabilities or roles; inference must bridge the gap. | Partially observable (multi-agent) decision problems; centralized training with decentralized execution (CTDE) when extra information exists only at training time; latent-type or latent-policy models in AHT. | **Information nerf:** reduced sight radius and hiding teammate level in Level-Based Foraging (LBF), so behavioral inference (e.g., GPL’s LSTM type vectors) becomes load-bearing. This axis is *directly* about what the learner can see. |
| **B. Non-stationarity of the composition process** | The rule that draws *which* teammates appear **across episodes** is not fixed; the learner may face a shifting mixture over types or policies even when within-episode dynamics are Markov given the drawn team. | Distinct threads: heterogeneous teams (agents differ), **open** composition (sets or types vary), **stationary** sampling (i.i.d. from a fixed mixture), versus **drift** (the mixture or population statistics evolve). | **Population drift:** an Ornstein–Uhlenbeck process on the type-frequency simplex advances at episode boundaries; compositions are sampled from the current mixture. |

**Framing discipline (axis B).** Heterogeneity—agents differing in level, skill, or objective—is almost always present in AHT and is a *prerequisite* for composition to matter. It is **not** the same object as drift. One can hold heterogeneity fixed (several discrete types) while making the **generating process** over those types **stationary** (standard open-AHT evaluation: random teammates i.i.d. from a fixed distribution[^rahman-jmlr]). Conversely, drift in this MQP is **not** “more heterogeneity” in the abstract; it is **temporal structure on the mixture** over types that were already in the training support. The motivating deployments (rotating personnel, shifting skill mixes, fleets replaced over quarters) are stories about **how often** each type appears, not necessarily about inventing wholly new types. That distinction matters when connecting to prior work on *open-set* generalization (novel types never seen at training), which this project explicitly does not emphasize.

**Framing discipline (axis A).** Partial observability here is **operational**: we restrict sensing (sight, level visibility) so that the same GPL machinery must infer latent structure from trajectories. That is aligned with POMDP-style difficulty but **does not** by itself implement drift; it sharpens the need for type inference so that downstream questions about robustness are non-vacuous.

---

## 3. Axis A—Partial Observability, Latent Teammates, and GPL-Class Methods

GPL (Graph-based Policy Learning) represents the state of the art for **open** AHT under variable composition: an LSTM produces continuous type embeddings, a relational model predicts teammate actions, and a coordination-graph Q-factor selects the learner’s action—supporting both full and reduced observability depending on environment features.[^rahman-icml][^rahman-jmlr] ODITS removes reliance on a small finite type set by learning latent teammate structure **online** under partial observability—closer to the “decode the team from what you see” problem this MQP foregrounds when levels are hidden.[^gu] CIAO reframes open AHT with cooperative game theory and offers theoretical grounding for graph-structured value decompositions akin to GPL’s coordination graph.[^wang-ciao]

Privileged information at training time (CTDE) appears across MARL; in AHT it supports auxiliary objectives that shape teammate models without requiring privileged fields at execution. ODITS’s decoding-style losses are one example; our later auxiliary-level head (see research document §4) sits in the same family—predicting a discrete capability from embeddings—rather than claiming a new representational paradigm.

**Fast teammate adaptation (Fastap).** Zhang and colleagues study **abrupt** change in a teammate’s *policy* and rapid adaptation—non-stationarity at the behavioral level within the AHT framing.[^zhang-fastap] That complements, and should not be conflated with, **mixture drift** over types across episodes: Fastap targets a different time scale and a different latent (policy identity versus population frequency).

---

## 4. Axis B—Composition, Populations, and Stationarity in Prior Work

**Generalization across compositions, still stationary.** COPA uses a coach–player hierarchy and training over a predefined set of team layouts to zero-shot generalize to held-out layouts from the **same** discrete family.[^liu-copa] The emphasis is spatial or roster structure, not a continuously drifting **frequency** over types through time.

**Population- and policy-aware extensions.** Recent N-agent ad hoc teamwork (NAHT) and the POAM algorithm broaden who is controlled and emphasize learning teammate behavior representations for out-of-distribution teammates at evaluation—again advancing composition flexibility, with evaluation still typically embedded in **stationary** test protocols unless authors add explicit temporal drift.[^wang-poam]

**Other non-stationary AHT/MARL threads.** Santos and colleagues study AHT when individual teammates are **non-stationary** in the sense of changing behavior over time.[^santos] Mao and colleagues give model-free non-stationary RL regret theory with multi-agent applications—valuable for abstract non-stationarity, not a direct model of simplex mixture drift.[^mao] These works justify treating “stationary i.i.d. teammates” as a special case rather than the only realistic case; they do not replace a **parameterized drift model** over population mixtures for the question we ask here.

---

## 5. Synthesis: Why Both Axes Appear in This Project

GPL’s published LBF evaluation uses observations rich enough that type inference is largely bypassable; under that regime, **axis B** (drift over mixtures) barely stresses the algorithm because **axis A** never forces the policy to depend on inferred types. The MQP therefore **orthogonalizes** the story: (1) an information-nerfed regime makes **axis A** real; (2) OU drift on the mixture implements **axis B** as a controlled, smooth, temporally correlated departure from episode-i.i.d. stationarity. Neither axis “substitutes” for the other: drift does not create partial observability, and hiding levels does not model a drifting population—**together** they ask whether graph-based open-AHT learning remains useful when both difficulties apply.

---

## Notes

[^stone]: Peter Stone, Gal A. Kaminka, Sarit Kraus, and Jeffrey S. Rosenschein, “Ad Hoc Autonomous Agent Teams: Collaboration without Pre-Coordination,” in *Proceedings of the 24th AAAI Conference on Artificial Intelligence* (2010), 1504–9.

[^mirsky]: Reuth Mirsky et al., “A Survey of Ad Hoc Teamwork Research,” in *Multi-Agent Systems: EUMAS 2022*, Springer LNCS (2022), 275–93, arXiv:2202.10450.

[^rahman-icml]: Md. Ashiqur Rahman et al., “Towards Open Ad Hoc Teamwork Using Graph-Based Policy Learning,” in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139 (2021), 8776–86.

[^rahman-jmlr]: Arrasy Rahman et al., “A General Learning Framework for Open Ad Hoc Teamwork Using Graph-Based Policy Learning,” *Journal of Machine Learning Research* 24, no. 298 (2023): 1–74.

[^gu]: Peihao Gu et al., “Online Ad Hoc Teamwork under Partial Observability,” in *Proceedings of the International Conference on Learning Representations (ICLR)* (2022).

[^wang-ciao]: Jiachen Wang et al., “Open Ad Hoc Teamwork with Cooperative Game Theory,” in *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235 (2024), 50902–30, arXiv:2402.15259.

[^liu-copa]: Boyuan Liu et al., “Coach-Player Multi-Agent Reinforcement Learning for Dynamic Team Composition,” in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139 (2021), arXiv:2105.08692.

[^wang-poam]: Caroline Wang et al., “N-Agent Ad Hoc Teamwork,” *Advances in Neural Information Processing Systems* 37 (NeurIPS 2024), arXiv:2404.10740.

[^santos]: Pedro M. Santos et al., “Ad Hoc Teamwork in the Presence of Non-Stationary Teammates,” in *Progress in Artificial Intelligence: EPIA 2021*, Springer LNCS 12981 (2021), 648–60.

[^zhang-fastap]: Ziqian Zhang et al., “Fast Teammate Adaptation in the Presence of Sudden Policy Change,” in *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 216 (2023), 2465–76, arXiv:2305.05911.

[^mao]: Weichao Mao et al., “Model-Free Nonstationary Reinforcement Learning: Near-Optimal Regret and Applications in Multiagent Reinforcement Learning and Inventory Control,” *Management Science* 71, no. 2 (2024): 1564–80.

---

## Bibliography

Entries follow the bibliography style of Kate L. Turabian, *A Manual for Writers of Research Papers, Theses, and Dissertations*, 9th ed. (Chicago: University of Chicago Press, 2018), aligned with *The Chicago Manual of Style*—author-date elements omitted here in favor of a single alphabetical **References** list suitable for a thesis bibliography.

Gu, Peihao, Minguk Zhao, Jianye Hao, and Bo An. “Online Ad Hoc Teamwork under Partial Observability.” In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2022.

Liu, Boyuan, Qi Liu, Peter Stone, Anima Garg, Yuke Zhu, and Anima Anandkumar. “Coach-Player Multi-Agent Reinforcement Learning for Dynamic Team Composition.” In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 2021. arXiv:2105.08692.

Mao, Weichao, Kaiqing Zhang, Ruihao Zhu, David Simchi-Levi, and Tamer Başar. “Model-Free Nonstationary Reinforcement Learning: Near-Optimal Regret and Applications in Multiagent Reinforcement Learning and Inventory Control.” *Management Science* 71, no. 2 (2024): 1564–80.

Mirsky, Reuth, Ignacio Carlucho, Arrasy Rahman, Elise Fosong, Wendy Macke, Mohan Sridharan, Peter Stone, and Stefano V. Albrecht. “A Survey of Ad Hoc Teamwork Research.” In *Multi-Agent Systems: EUMAS 2022*, edited by Nils Bulling and Amro Najjar, 275–93. Cham: Springer, 2022. arXiv:2202.10450.

Rahman, Arrasy, Ignacio Carlucho, Niklas Höpner, and Stefano V. Albrecht. “A General Learning Framework for Open Ad Hoc Teamwork Using Graph-Based Policy Learning.” *Journal of Machine Learning Research* 24, no. 298 (2023): 1–74.

Rahman, Md. Ashiqur Rahman, Niklas Höpner, Filippos Christianos, and Stefano V. Albrecht. “Towards Open Ad Hoc Teamwork Using Graph-Based Policy Learning.” In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 8776–86, 2021.

Santos, Pedro M., João G. Ribeiro, Ana Sardinha, and Francisco S. Melo. “Ad Hoc Teamwork in the Presence of Non-Stationary Teammates.” In *Progress in Artificial Intelligence*, edited by Luís Soares Barbosa and Luis Farinas del Cerro, 648–60. Cham: Springer, 2021.

Stone, Peter, Gal A. Kaminka, Sarit Kraus, and Jeffrey S. Rosenschein. “Ad Hoc Autonomous Agent Teams: Collaboration without Pre-Coordination.” In *Proceedings of the 24th AAAI Conference on Artificial Intelligence*, 1504–9. AAAI Press, 2010.

Wang, Caroline, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone. “N-Agent Ad Hoc Teamwork.” In *Advances in Neural Information Processing Systems* 37, 2024. arXiv:2404.10740.

Wang, Jiachen, Yanchen Li, Yifeng Zhang, Wei Pan, and Samuel Kaski. “Open Ad Hoc Teamwork with Cooperative Game Theory.” In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 50902–30, 2024. arXiv:2402.15259.

Zhang, Ziqian, Lei Yuan, Lu Li, Kai Xue, Chenhao Jia, Cong Guan, Chun Qian, and Yang Yu. “Fast Teammate Adaptation in the Presence of Sudden Policy Change.” In *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence*, PMLR 216, 2465–76, 2023. arXiv:2305.05911.
