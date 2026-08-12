# Modules of a Federated Learning System: What to Build and Which Options Exist

This document breaks a Federated Learning (FL) system into its independent, swappable modules — like a parts list for building or evaluating an FL platform (whether you're building it in PyTorch, Flower, TensorFlow Federated, or from scratch). For each module: **what it does, why it exists, and the concrete implementation options**, ranging from simplest to most advanced.

---

## System overview (how the modules connect)

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                         SERVER / ORCHESTRATOR                      │
 │                                                                     │
 │  [Client Registry] → [Client Selector] → [Round Orchestrator]      │
 │        ↑                                        │                  │
 │  [Reputation /                                   ▼                  │
 │   Incentive Module]                    [Communication Module]      │
 │        ↑                                        │                  │
 │  [Monitoring / Logging] ←──────── [Aggregation Module] ←───────┐   │
 │                                          │                     │   │
 │                                   [Global Model Store]         │   │
 └───────────────────────────────────────────────────────────────┼───┘
                                    │                             │
                    (send model)   ▼                (send update) │
 ┌───────────────────────────────────────────────────────────────┴───┐
 │                              CLIENT                                 │
 │  [Local Data Module] → [Local Trainer] → [Compression/Privacy]      │
 │                              ↑                                      │
 │                    [Personalization Module]                         │
 └───────────────────────────────────────────────────────────────────┘
```

---

## 1. Client Registry / Membership Module
**Purpose:** Keep track of which clients exist, their metadata (capabilities, availability windows, last-seen time), and whether they are eligible to participate.

**Implementation options:**
- **Static registry** — a fixed list/config file of client IDs (simplest; fine for research prototypes and cross-silo settings with few, known clients).
- **Dynamic registry with heartbeat/check-in** — clients periodically ping the server (used in cross-device settings like Google's production FL, where millions of phones join/leave); implemented via a lightweight key-value store (Redis) or a dedicated FL platform's built-in device manager.
- **Capability-tagged registry** — stores device compute class, network type (WiFi/cellular), battery state, historical reliability score alongside the ID, feeding directly into the Client Selector module.

---

## 2. Client Selector Module
**Purpose:** Decide, each round, which subset of registered/eligible clients will participate. This is the module covered in depth in Documents 1 and 3.

**Implementation options (from simplest to most sophisticated):**
- **Uniform random sampling** — `random.sample(clients, K)`; the FedAvg default. Simple, unbiased, ignores everything about client quality.
- **Data-size-weighted sampling** — sample with probability `p_i = n_i / Σ n_j`. Still unbiased, slightly better use of large clients.
- **Resource/deadline-constrained selection (FedCS-style)** — solve (greedily) "pick the max clients that finish within deadline `T`" using reported compute/network estimates.
- **Utility/loss-based selection (Power-of-Choice, Active FL)** — sample a candidate pool, probe local loss, keep the highest-loss subset.
- **Bandit-based adaptive selection (Oort, UCB-CS)** — maintain a running utility estimate per client and use an exploration–exploitation rule (e.g., ε-greedy or UCB) to pick clients over time.
- **Fairness-constrained selection (AFL, q-FFL-aligned)** — bias selection toward currently worst-performing clients, or enforce a minimum selection frequency per client over a sliding window.
- **Entropy/heterogeneity-aware selection (FedEntOpt, HiCS-FL)** — greedily pick clients that maximize label-distribution entropy of the selected batch, or cluster clients by estimated heterogeneity and sample across clusters.
- **Reputation/incentive-weighted selection (MURIM-style)** — combine a multidimensional reputation score (reliability + data value + fairness credit) into the selection probability.
- **Mobility/availability-predictive selection (MACS-style)** — forecast near-future connectivity per client (e.g., simple Markov model or learned predictor) and avoid selecting clients likely to disconnect mid-round.
- **Learned/agentic selection** — an RL policy or LLM agent that adapts the selection rule itself based on observed round outcomes.

**Practical note:** the selector should be a pluggable interface — `select(round_t, candidate_pool, client_stats) → subset` — so different strategies from Document 3's table can be swapped without touching the rest of the system.

---

## 3. Round Orchestrator Module
**Purpose:** Drive the overall training loop — trigger rounds, set deadlines, decide synchronous vs. asynchronous aggregation, handle timeouts/dropouts.

**Implementation options:**
- **Synchronous rounds (FedAvg default)** — server waits for all `K` selected clients (or a deadline) before aggregating. Simple, but round time = slowest responder.
- **Asynchronous FL** — server aggregates updates as they arrive, weighting older ("stale") updates less (staleness-aware weighting, e.g., a decay factor `α(τ)` where `τ` is staleness in rounds). Better for highly heterogeneous, unreliable clients; more complex convergence analysis.
- **Semi-synchronous / bucketed rounds** — group clients into buckets by expected response time and aggregate bucket-by-bucket (a middle ground, used in some hierarchical/edge FL systems).
- **Deadline + partial aggregation** — proceed with whichever selected clients respond by a cutoff time, common with FedCS-style selection.

---

## 4. Communication Module
**Purpose:** Move the model and updates between server and clients efficiently and (optionally) securely.

**Implementation options:**
- **Plain parameter transfer** — send/receive full model weights each round (simplest, most bandwidth-heavy).
- **Compression — quantization** — reduce parameter precision (e.g., FP32 → INT8, or stochastic quantization); can be adaptive per client based on bandwidth (as in GRACE-FL-style green/edge-aware FL, reporting up to ~75% overhead reduction on constrained hardware).
- **Compression — sparsification** — send only the top-k largest-magnitude updates per client, with a residual/error-feedback buffer to accumulate what was dropped.
- **Structured/low-rank updates** — restrict local updates to a low-rank or structured subspace, cutting the number of values transmitted.
- **Secure channels** — TLS for transport security; this is separate from, and in addition to, any cryptographic privacy mechanism (Module 8).
- **Protocol choice** — gRPC (common in Flower, TFF), REST/HTTP for simpler setups, MQTT for constrained IoT deployments.

---

## 5. Local Trainer Module (runs on each client)
**Purpose:** Perform the actual local model update on-device.

**Implementation options:**
- **Standard local SGD (FedAvg)** — run `E` local epochs of mini-batch SGD from the received global weights.
- **FedProx-style local training** — add a proximal term `(μ/2)‖w − w^t‖²` to the local loss to keep local updates from drifting too far under non-IID data — a direct complement to whatever the Selector does.
- **SCAFFOLD-style control-variate training** — maintain and exchange control variates to explicitly correct for client drift, orthogonal to which clients are selected.
- **Differentially private local training (DP-SGD)** — clip and noise gradients locally before they ever leave the device (belongs jointly to Module 5 and Module 8).

---

## 6. Aggregation Module (runs on server)
**Purpose:** Combine the returned client updates into a new global model.

**Implementation options:**
- **FedAvg (weighted mean)** — `w^{t+1} = Σ q_i w_i^{t+1}`, `q_i` typically data-size-proportional.
- **Simple (unweighted) mean** — equal weight per client, useful when data-size weighting itself would introduce unwanted bias toward large clients.
- **Trimmed mean / median aggregation** — coordinate-wise trimmed mean or median across client updates; a lightweight Byzantine-robustness measure.
- **Krum** — pick the single update whose sum of squared distances to its nearest neighbors is smallest (robust to up to `f` malicious clients, assuming `n ≥ 2f + 3`).
- **Bulyan** — Krum-style shortlist, then trimmed-mean aggregation over the shortlist; stronger Byzantine robustness than Krum alone.
- **Secure aggregation (cryptographic)** — clients' updates are summed via secret-sharing/masking so the server only ever sees the *sum*, never an individual client's update (protects privacy even from the server).

---

## 7. Personalization Module (optional, client-side or server-side)
**Purpose:** Adapt the shared global model to each client's own distribution, since one global model is rarely optimal for every client under non-IID data.

**Implementation options:**
- **Fine-tuning** — each client fine-tunes the received global model on its own data for a few extra local steps before use (simplest).
- **Meta-learning-based personalization (Per-FedAvg)** — train the global model so that it is a good *starting point* for one-step client-specific adaptation (MAML-style).
- **Multi-task / clustered personalization** — group clients into clusters (often reusing Module 2's heterogeneity clustering) and train one model per cluster instead of one global model.
- **Model mixing / interpolation** — client keeps a local model and mixes it with the received global model, e.g. `w_i = λ w_local + (1-λ) w_global`.

---

## 8. Privacy & Security Module
**Purpose:** Protect data confidentiality and defend against malicious participants.

**Implementation options:**
- **Differential Privacy (DP)** — add calibrated noise to updates (locally, i.e., "local DP," or at the server after secure aggregation, i.e., "central DP"), with a formal `(ε, δ)` privacy budget.
- **Secure Aggregation** — cryptographic protocol (e.g., pairwise masking) so raw individual updates are never revealed to the server, only their sum.
- **Homomorphic encryption** — allows the server to aggregate encrypted updates directly, at higher computational cost, useful in cross-silo settings with strong trust requirements.
- **Byzantine-robust aggregation** — Krum, Bulyan, trimmed mean (see Module 6); defends against malicious/corrupted clients rather than an honest-but-curious server.
- **Anomaly/outlier detection on updates** — statistical or learned detectors flag suspicious updates (e.g., unusually large norm) before they reach aggregation.

---

## 9. Incentive & Reputation Module
**Purpose:** Encourage honest, sustained participation, especially in cross-silo/business FL where clients bear real resource costs.

**Implementation options:**
- **Flat/fixed payment per round** — simplest; does not account for contribution quality.
- **Contribution-based payment (e.g., Shapley-value approximation)** — pay clients proportional to their estimated marginal contribution to model improvement.
- **Reputation scoring** — maintain a running score per client based on reliability, data quality signals, and past behavior; feed this score back into Module 2 (Selector) as a selection-probability multiplier.
- **Multidimensional reputation with fairness correction (MURIM-style)** — jointly track privacy risk, resource contribution, and reward fairness, with an explicit mechanism (e.g., a "rare-direction" boost) to avoid the reputation system itself reinforcing majority-client bias.
- **Blockchain-recorded contribution ledger** — for decentralized/trust-minimized settings, record contributions and rewards on-chain for auditability.

---

## 10. Monitoring, Logging & Evaluation Module
**Purpose:** Track round-level and client-level metrics for debugging, research reporting, and feeding adaptive modules (Selector, Incentive).

**Implementation options:**
- **Basic metrics logging** — loss/accuracy per round, wall-clock round time, number of successful/dropped clients (simplest; e.g., TensorBoard, Weights & Biases).
- **Per-client dashboards** — track individual client selection frequency, contribution quality, and fairness metrics (variance of per-client accuracy) over time.
- **Communication/energy accounting** — log bytes transferred and estimated energy per client per round, useful for resource-aware and "green FL" objectives.
- **Federated evaluation** — run held-out evaluation on client devices themselves (rather than a centrally held test set) via a "testing selector" (as in Oort), to respect the no-raw-data-leaves-device constraint even for evaluation.

---

## 11. Topology Module (only for Decentralized/P2P FL)
**Purpose:** In server-less FL, define which clients talk to which, replacing the classic server-centric round loop.

**Implementation options:**
- **Fixed graph topology** — a pre-defined static communication graph (e.g., ring, random graph).
- **Gossip protocol** — clients periodically exchange and average updates with a randomly chosen neighbor.
- **Blockchain-backed coordination** — updates and contribution records posted on a distributed ledger, giving auditability without a trusted central party.
- **Trust/topology-aware neighbor selection** — analogous to Module 2, but selecting *neighbors* rather than server-side clients, weighted by trust/reputation and network proximity.

---

## Minimal vs. production-grade module checklist

| Module | Minimal prototype (research code) | Production system |
|---|---|---|
| Client Registry | static list | dynamic heartbeat registry |
| Client Selector | uniform random | utility/bandit-based + fairness constraints |
| Orchestrator | synchronous, fixed round count | async or semi-sync with deadlines |
| Communication | plain weight transfer | quantized/sparsified, TLS |
| Local Trainer | plain SGD | FedProx/SCAFFOLD + optional DP-SGD |
| Aggregation | FedAvg mean | robust aggregation (trimmed mean/Bulyan) + secure aggregation |
| Personalization | none | fine-tuning or clustered personalization |
| Privacy/Security | none | DP + secure aggregation + anomaly detection |
| Incentives | none | reputation-weighted rewards |
| Monitoring | console logs | full dashboards + federated evaluation |
| Topology (if P2P) | n/a | gossip + trust-weighted neighbor selection |

---

## Notes on frameworks that implement several modules out of the box
- **Flower (`flwr`)**: pluggable strategy interface maps directly onto Modules 2, 3, and 6 (custom `Strategy` classes implement selection + aggregation).
- **TensorFlow Federated (TFF)**: strong support for Modules 6 (aggregation) and 8 (DP via `tff.aggregators`).
- **FedML / NVIDIA FLARE**: broader coverage including Modules 9–10 (incentives, monitoring) and cross-silo orchestration.

These are mentioned only as reference points for where each module typically lives in existing tooling — the module boundaries above are framework-agnostic and apply whether you build from scratch or extend an existing platform.
