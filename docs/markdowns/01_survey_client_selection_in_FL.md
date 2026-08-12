# Client Selection in Federated Learning: A Self-Contained Survey

## Abstract

Federated Learning (FL) trains a shared model across many clients (phones, hospitals, IoT devices, organizations) without moving raw data to a central server. In every training round the server can normally only afford to work with a small fraction of the available clients. **Which** clients get picked in that fraction — the *client selection problem* — turns out to control almost everything that matters in practice: how fast the model converges, how accurate the final model is, how fair it is across clients, how much it costs in time and energy, and how robust it is to attackers. This document is a self-contained, mathematically grounded survey of client selection. It defines the FL optimization problem precisely, explains *why* naive (uniform random) selection is sub-optimal, builds a taxonomy of selection strategies grounded in a systematic literature review of 47 primary studies, walks through the convergence theory that justifies biased selection, and closes with open problems. No external reading is required to follow the mathematics; every symbol is defined where it is introduced, and a glossary is provided at the end.

---

## 1. Introduction

### 1.1 What Federated Learning is, in one paragraph

Federated Learning was introduced by McMahan et al. (2017) as a way to train a model on data that never leaves the device that generated it. Instead of the classical pipeline "collect data → centralize it → train," FL flips the order: the model travels to the data. A central server holds a global model, sends a copy to a set of clients, each client trains it briefly on its own local data, and the clients send back only the *updated model parameters* (not the data) for the server to combine ("aggregate") into a new global model. This is repeated for many rounds.

### 1.2 Why client selection is a separate, hard problem

A production FL deployment (e.g., a keyboard-prediction model running on millions of phones) cannot contact every device in every round — devices are offline, on metered connections, low on battery, or simply too numerous to coordinate. So each round the server selects a small subset of size `K` out of a population of `N` clients, where `K ≪ N`. The original FL algorithm (FedAvg) selects this subset **uniformly at random**. This is simple and statistically unbiased, but it interacts badly with four realities of deployed FL:

1. **Statistical heterogeneity** — client data is *non-IID* (not independent and identically distributed): a hospital in one city sees different disease prevalence than another; a user's typing style differs from the population average.
2. **System heterogeneity** — clients differ wildly in compute speed, memory, network bandwidth, and availability.
3. **Communication cost** — every selected client must download and upload a model, which is often the true bottleneck, not computation.
4. **Fairness and incentives** — if only fast, always-on, data-rich clients are ever selected, the model overfits to them and other clients receive no benefit (and have no reason to keep participating).

A 2023 systematic literature review of 47 primary studies on this topic found that these four challenge categories — **heterogeneity, resource allocation, communication cost, and fairness** — account for the overwhelming majority of published client-selection research, with heterogeneity alone the focus of roughly half of all reviewed studies. This survey is organized around that same challenge structure.

---

## 2. Formal Problem Setup

We build up the mathematics in layers so every symbol is grounded before it is used.

### 2.1 The learning objective

There are `N` clients, indexed `i = 1, …, N`. Client `i` holds a local dataset `D_i` of size `n_i`, drawn from a local (possibly client-specific) data distribution. Client `i`'s local objective (e.g., average loss over its own data) is

```
F_i(w) = (1 / n_i) * Σ_{(x,y) ∈ D_i} ℓ(w; x, y)
```

where `w` is the model's parameter vector and `ℓ` is a per-example loss function (e.g., cross-entropy).

The **global objective** that FL tries to minimize is a weighted average of the local objectives:

```
F(w) = Σ_{i=1}^{N} p_i * F_i(w),      where p_i = n_i / Σ_j n_j
```

so `p_i` is client `i`'s share of the total data, and the weights `p_i` sum to 1. The FL goal is `w* = argmin_w F(w)`.

### 2.2 FedAvg with partial participation

In round `t`, the server holds global weights `w^t`. It selects a subset `S_t ⊆ {1,…,N}` of size `K`. Each selected client runs `E` local epochs of (stochastic) gradient descent starting from `w^t`, producing a local update `w_i^{t+1}`. The server aggregates:

```
w^{t+1} = Σ_{i ∈ S_t} q_i * w_i^{t+1}
```

where `q_i` are aggregation weights (commonly `q_i = n_i / Σ_{j∈S_t} n_j`, i.e., data-size-weighted averaging — this is standard FedAvg).

### 2.3 What makes a selection scheme "unbiased"

A selection scheme is **unbiased** if the *expected* aggregated update equals what you would get from a full-participation round:

```
E_{S_t}[ (1/K) * Σ_{i∈S_t} g_i ]  =  (1/N) * Σ_{i=1}^{N} g_i
```

where `g_i` denotes client `i`'s local gradient (or update). Uniform random sampling without replacement, and sampling with probability proportional to `n_i` (data size), are the two classic unbiased schemes. Any scheme that *systematically* favors certain clients (fast clients, high-loss clients, clients with rare labels) is **biased** — biased schemes can converge faster but converge to a solution that is subtly skewed toward the favored clients unless the bias is corrected for (e.g., via importance-weighted aggregation).

### 2.4 Why uniform random selection is provably sub-optimal

Cho, Wang & Joshi (2020, 2022) give the first convergence analysis that explicitly compares biased against unbiased client selection. Their key result, stated informally:

> Biasing selection toward clients with **higher current local loss** `F_i(w^t)` strictly increases the guaranteed rate of convergence, at the cost of converging to a solution with a bounded bias term that shrinks as the selection becomes less aggressive.

This single insight — *select where the model is currently doing worst* — is the theoretical seed for an entire family of practical algorithms (Section 4.3). The practical algorithm that operationalizes it is called **Power-of-Choice**:

1. **Candidate sampling:** draw a candidate pool `A` of size `d` (with `K ≤ d ≤ N`) using data-size-proportional sampling.
2. **Loss probing:** query (or estimate) the current local loss `F_i(w^t)` for every client in `A`.
3. **Greedy pick:** select the `K` clients in `A` with the *highest* local loss.

As `d → N` the bias (and the convergence speed-up) grows; as `d → K` the scheme degenerates back to unbiased random sampling. This `d` parameter is therefore a **direct knob on the speed/fairness trade-off**, and nearly every later "smarter" selection method can be understood as a different way of estimating "which clients would most help right now" instead of using raw loss.

---

## 3. Why Random Selection Fails in Practice: The Four Challenges

### 3.1 Statistical heterogeneity (non-IID data)

When client data distributions differ, a round that happens to sample clients whose data over-represents certain classes produces an update that pulls the global model away from the true optimum — a phenomenon sometimes called *client drift*. This is empirically the single most-studied challenge in the client-selection literature (found in roughly half of the 47 studies reviewed in the 2023 SLR). Mitigations include entropy-maximizing selection (Section 4.5) and heterogeneity-aware clustering (Section 4.6).

### 3.2 System heterogeneity (resource allocation)

Clients differ in CPU/GPU speed, RAM, and battery. If the server always waits for the slowest selected client ("straggler"), round time is dictated by the weakest link. FedCS (Section 4.2) was the first to explicitly formalize this as a **deadline-constrained selection problem**.

### 3.3 Communication cost

Uploading/downloading full model parameters every round is expensive on cellular/edge networks. This motivates (a) selecting fewer, more valuable clients per round, and (b) compressing what selected clients send (quantization, sparsification) — a modeling choice at the intersection of client selection and the *communication module* discussed in Document 2.

### 3.4 Fairness and incentives

If selection always favors clients that are fast, well-connected, or high-utility, two problems emerge: (a) the model underperforms for the systematically excluded clients (a *good-intent fairness* violation), and (b) rationally self-interested clients (in a cross-silo, business setting) have no incentive to keep contributing resources if they are rarely selected or never rewarded. This motivates incentive-aware and reputation-based selection (Section 4.8).

---

## 4. A Taxonomy of Client Selection Strategies

We group strategies by *what signal they optimize for*. (Document 3 gives the full formula-by-formula catalog; this section explains the ideas and connects them into families.)

### 4.1 Random / statistical baselines
Uniform random sampling and data-size-proportional sampling. Unbiased, cheap, but blind to everything in Section 3. Serves as the control group against which every other method is benchmarked.

### 4.2 Resource-aware (system-aware) selection
Optimizes for round *speed* and *feasibility* given device/network constraints.
- **FedCS** (Nishio & Yonetani, 2019): clients report an estimated round-completion time; the server greedily selects the maximum number of clients that can finish (download → local update → upload) before a fixed deadline. This is a knapsack-style resource allocation problem, solved with a greedy heuristic. FedCS is the foundational resource-constrained selector, but it can be biased toward high-end devices, whose data may not represent the population well.
- **GRACE-FL style approaches**: extend deadline-based selection with adaptive compression (e.g., quantization) chosen per client based on its bandwidth/energy budget, reported to cut communication overhead substantially (up to ~75% in on-device edge evaluations) while preserving accuracy.
- **Mobility-aware selection (MACS-type methods)**: for vehicular/IoT clients, predict *future* connectivity/mobility so the server avoids selecting clients likely to drop out mid-round, reducing wasted rounds.

### 4.3 Performance / utility-aware selection
Optimizes for *statistical* contribution to model improvement.
- **Power-of-Choice** (Section 2.4): select the highest-current-loss clients from a random candidate pool.
- **Active Federated Learning**: treats client value as a function of loss, similar in spirit to Power-of-Choice, using it to bias a softmax-style selection probability rather than a hard top-K cut.
- **Oort** (Lai et al., 2021): defines a client **utility score** that combines (a) the statistical utility of the client's data (how much its gradient reduces global loss, approximated from training loss statistics) and (b) the client's expected round duration (system speed). Formally, Oort favors clients maximizing something like
  ```
  Utility_i ∝ (statistical utility of client i) × (system speed penalty of client i)
  ```
  and uses an **exploration–exploitation** strategy borrowed from multi-armed bandits: mostly exploit known high-utility clients, but keep sampling under-explored ones so that new or currently-idle high-value clients are not permanently ignored, and so that utility estimates do not go stale. Reported to improve time-to-accuracy by roughly 1.2×–14× and final accuracy by 1–10 percentage points versus random/resource-only baselines.
- **UCB-CS (bandit-based selection)**: casts client selection explicitly as a multi-armed-bandit problem — each client is an "arm," reward is the observed contribution to loss reduction, and an Upper-Confidence-Bound rule balances trying new clients against exploiting known-good ones.

### 4.4 Fairness-aware selection
- **Agnostic Federated Learning (AFL)**: instead of minimizing the *average* loss across clients, minimizes the *worst-case* client loss (a min-max formulation), which the server approximates by adaptively re-weighting/selecting clients whose loss is currently largest — related to, but philosophically distinct from, Power-of-Choice, because the goal is equity rather than raw speed.
- **q-Fair FL (q-FFL)**: generalizes the objective to `Σ p_i · F_i(w)^{q+1}/(q+1)`; larger `q` pushes the optimizer to care more about the worst-off clients, changing which clients "matter most" for selection/aggregation.
- **PHP-FL-type methods**: explicitly correct for *unequal participation probability* across rounds, so that clients who are structurally less likely to be online (and hence less likely selected) are not systematically under-represented in the final model.

### 4.5 Data-distribution-aware selection
- **FedEntOpt**: greedily builds the selected set round-by-round to **maximize the entropy of the aggregated label distribution** of the chosen clients, directly targeting label-distribution skew (a common form of non-IID-ness) without needing raw label statistics to be centrally stored — labels can be aggregated in a privacy-preserving count.
- **HiCS-FL**: estimates each client's data heterogeneity indirectly, from the **statistics of the last (output) layer's gradient/bias updates** returned during training, then clusters clients by estimated heterogeneity so that each round's sample better represents the full spread of the population instead of by-chance clustering around one mode.

### 4.6 Clustering-based selection
Groups clients (by data distribution, gradient similarity, or resource profile) and samples proportionally *across* clusters rather than from the raw population, guaranteeing coverage of minority modes of the data distribution that pure random sampling would frequently miss. Used in *clustered FL over wireless edge networks* variants.

### 4.7 Security- and robustness-aware selection
When some clients may be malicious (data poisoning, model poisoning, or simple faults), naive aggregation (plain averaging) is fragile — a single bad update can dominate the mean. Two classic **Byzantine-robust aggregation/selection** rules:
- **Krum**: for each client's update, compute the sum of squared distances to its `n − f − 2` closest neighboring updates (where `f` is the assumed maximum number of malicious clients); select the single update with the smallest such sum as the round's representative update (or use it to score/filter which updates to trust before averaging).
- **Bulyan**: a two-stage defense — first repeatedly apply a Krum-like rule to build a *shortlist* of the most trustworthy updates, then aggregate only that shortlist using a **coordinate-wise trimmed mean** (dropping the highest and lowest values per parameter before averaging), giving stronger guarantees than Krum alone against certain colluding-attacker strategies.

These are technically *aggregation* rules more than *selection* rules, but in the survey literature they are grouped with security-aware client selection because both address "which client contributions should count."

### 4.8 Incentive- and reputation-aware selection
Cross-silo/business FL (banks, hospitals, telecom operators) treats participation as a resource that must be *paid for*. Reputation-based incentive mechanisms (e.g., "MURIM"-style multidimensional reputation) jointly score clients on privacy risk, resource contribution, historical reliability, and *fairness of reward*, and explicitly boost weighting for clients whose data direction is statistically rare (to avoid the reputation system itself re-creating the majority-bias problem) — often implemented via something like a "subspace leverage" boost for under-represented gradient directions.

### 4.9 Decentralized / peer-to-peer and blockchain-based selection
Not all FL is server-centric. In **decentralized FL (DFL)**, there is no single aggregator; clients exchange updates directly with neighbors in a peer-to-peer topology (gossip protocols) or record contributions on a blockchain for auditability and trust when no party is fully trusted. "Client selection" here becomes **neighbor selection / topology design**: which peers to average with in each round, subject to network topology, trust scores, and communication-graph constraints. A recent survey of DFL (covering research trends from roughly 2018–2026) documents this shift and gives a unified challenge-driven taxonomy spanning connectivity, personalization, security, and topology-aware optimization, mirroring the centralized taxonomy above but adapted to a graph rather than a star topology.

### 4.10 Agentic / learned selection (emerging direction)
A newer line of work proposes using LLM-based agents to *design and adapt* the selection (and broader FL) strategy automatically — e.g., an agent observes round statistics and proposes changes to the selection policy, hyperparameters, or even code, rather than a human hand-tuning a fixed formula. This is promising but currently lacks the large-scale empirical validation of the methods in Sections 4.2–4.7.

---

## 5. Evaluation: How Client Selection Methods Are Actually Measured

Comparing methods requires standardized metrics. The main ones used across the literature:

| Metric | What it measures | Typical formula / definition |
|---|---|---|
| Time-to-accuracy (TTA) | Wall-clock time to reach a target accuracy | wall-clock rounds × per-round time, until `Acc(w^t) ≥ target` |
| Rounds-to-accuracy | Statistical efficiency, independent of hardware | min `t` such that `Acc(w^t) ≥ target` |
| Final accuracy / loss | Model quality at convergence | `F(w^T)` or held-out test accuracy |
| Fairness (variance of accuracy) | Spread of per-client accuracy | `Var_i[Acc_i(w^T)]` (lower = fairer) |
| Communication cost | Total bytes moved | `Σ_t Σ_{i∈S_t} (upload_i + download_i)` |
| Client coverage / participation equity | How evenly clients are chosen over time | e.g., Gini coefficient of per-client selection counts |
| Robustness | Accuracy degradation under `f` malicious clients | `Acc_clean − Acc_under_attack` |

A well-designed evaluation reports at least one metric from each *challenge category* (Section 3) rather than accuracy alone, since a method can win on accuracy while silently failing on fairness or robustness.

---

## 6. Open Problems and Future Directions

1. **Handling "unsuccessful" clients gracefully.** Most selection theory assumes a selected client always successfully returns an update. In practice clients drop out mid-round (dead battery, lost connection). How to select *robustly to expected dropout*, rather than reactively discarding failed rounds, remains under-explored and was flagged as a specific gap in the 2023 SLR.
2. **Joint optimization across challenge categories.** Most published methods optimize primarily for one axis (speed *or* fairness *or* robustness) and treat the others as secondary constraints. Multi-objective frameworks that jointly and provably balance all four challenge categories are still rare.
3. **Selection under differential privacy.** Adding per-round privacy noise interacts with *who* is selected and *how often* — biased selection strategies can silently violate the privacy-accounting assumptions baked into standard DP-FL analyses, requiring re-derivation of both the convergence bound and the privacy budget together.
4. **Decentralized/topology-aware selection at scale.** Peer-selection in leaderless, blockchain-audited FL is far less mature theoretically than the centralized case; convergence guarantees under adversarial or unreliable topologies are an open area.
5. **Agentic/automated policy design.** Using learned agents (including LLM-based ones) to design, adapt, or even generate new selection strategies at runtime, rather than relying on a fixed hand-designed rule, is a nascent but fast-moving direction.

---

## 7. Conclusion

Client selection is not a peripheral detail of Federated Learning — it is one of the few control points a system designer has over convergence speed, fairness, communication cost, and robustness simultaneously, precisely because only a fraction of clients participate in any given round. The field has converged on four recurring challenge categories (heterogeneity, resource allocation, communication cost, fairness) and a family of strategies that trade among them: resource-aware methods (FedCS-style) prioritize feasibility and speed; utility-aware methods (Power-of-Choice, Oort, bandit-based) prioritize statistical progress; fairness-aware and entropy/heterogeneity-aware methods prioritize representativeness and equity; and security-aware aggregation rules (Krum, Bulyan) prioritize robustness to malicious participants. No single method dominates on every axis — every published strategy is best understood as choosing a particular point on the speed–fairness–cost–robustness trade-off surface, and picking among them requires being explicit about which axis matters most for the deployment at hand.

---

## Glossary (for terms used above)

- **Client / participant**: a device or organization holding a private local dataset that participates in FL.
- **Round**: one iteration of send-model → local-train → return-update → aggregate.
- **Non-IID data**: client data distributions differ from each other and from the global population distribution.
- **Straggler**: a selected client that is unusually slow to complete and return its update.
- **Byzantine client**: a client that may send arbitrary (possibly malicious or corrupted) updates.
- **Bandit (multi-armed bandit)**: a sequential decision framework for balancing "explore new options" against "exploit known-good options."
- **Cross-device vs. cross-silo FL**: cross-device = huge numbers of small, unreliable clients (phones); cross-silo = few, large, reliable clients (hospitals, companies).

## References (representative)

- McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Nishio, T. & Yonetani, R. (2019). *Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge.* IEEE ICC.
- Cho, Y. J., Wang, J., & Joshi, G. (2020/2022). *Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies* / *Towards Understanding Biased Client Selection in Federated Learning.* arXiv:2010.01243; AISTATS 2022.
- Lai, F., Zhu, X., Madhyastha, H. V., & Chowdhury, M. (2021). *Oort: Efficient Federated Learning via Guided Participant Selection.* OSDI.
- Smestad, C. & Li, J. (2023). *A Systematic Literature Review on Client Selection in Federated Learning.* EASE.
- Blanchard, P. et al. (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum).* NeurIPS.
- Guerraoui, R. et al. (2018). *The Hidden Vulnerability of Distributed Learning in Byzantium (Bulyan).* ICML.
- Mohri, M., Sivek, G., & Suresh, A. T. (2019). *Agnostic Federated Learning.* ICML.
- Li, T. et al. (2020). *Fair Resource Allocation in Federated Learning (q-FFL).* ICLR.
- Chen, H. & Vikalo, H. (2024). *HiCS-FL: Heterogeneity-Guided Client Sampling for Federated Learning.*
- A comprehensive survey on client selection strategies in Federated Learning, *Computer Networks* (2024).
- A Comprehensive Survey On Client Selections in Federated Learning, arXiv:2311.06801.
- A Survey on Decentralized Federated Learning (2018–2026 trend survey).
