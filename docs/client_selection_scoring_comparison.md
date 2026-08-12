# Comparative Summary: Client Selection & Scoring Approaches in Federated Learning

This document summarizes the **client selection and scoring methodology** used in three papers, to support the design of an improved scoring-based client selection scheme.

---

## Paper 1: Dynamic Scoring-Based Client Selection for Diabetes Diagnosis
**Ahmed et al., Knowledge-Based Systems (2025)**

### Core Idea
A **weighted linear scoring function** ranks clients each round based on three metrics: accuracy, execution time, and loss. The top-k (50%) highest-scoring clients are selected per round.

### Scoring Formula
```
Score_i = α · accuracy_i − β · Normalized_Execution_Time_i − γ · Normalized_Loss_i
```
- α, β, γ = tunable weights for accuracy, execution time, and loss respectively
- Initial values: α = 0.5, β = 0.3, γ = 0.2

### Normalization
- **Execution time** (min-max, inverted so lower time → higher score):
  `norm_time = (max(t) − t) / (max(t) − min(t))`
- **Loss** (min-max, inverted so lower loss → higher score):
  `norm_loss = (max(L) − L) / (max(L) − min(L))`

### Dynamic Weight Adjustment
Weights are updated every round using a **3-round moving average trend**:
- If accuracy trend ↑ → α += δ, else α −= δ
- If execution time trend ↑ (getting worse) → β += δ
- If loss trend ↑ (getting worse) → γ += δ
- (δ = 0.01, a fixed learning rate for weight updates)

This lets the scoring system self-adjust emphasis toward whichever metric is currently problematic (e.g., penalize slow clients more if execution time worsens).

### Selection Mechanism
1. Compute score for **all** clients each round.
2. Sort clients descending by score.
3. Select **top k = N/2** clients (static selection *rate*, but *identity* of selected clients changes dynamically).
4. Optional **execution-time penalty** term if a client exceeds a max-time threshold.
5. Selected clients train locally (Random Forest); updates aggregated via FedAvg.

### Other Scoring Criteria Discussed (not all formally used in the formula)
The paper also *lists* — but does not mathematically incorporate — additional criteria that a more advanced scoring system could include:
- Data quality/heterogeneity, convergence speed, resource availability (CPU/memory/bandwidth), contribution consistency (historical performance), fairness/participation balance, communication cost, client dropout probability, impact on global model performance, and "previous record."

### Strengths
- Simple, interpretable, adaptive weighting.
- Directly ties client score to three measurable, cheap-to-compute metrics.
- Empirically validated over 200 rounds on a real healthcare dataset (BRFSS2015 diabetes data), 10 clients, achieving 0.83 accuracy.

### Limitations / Gaps
- No fairness constraint — pure top-k selection by score can systematically starve low-scoring (but potentially data-diverse) clients.
- No mechanism for adversarial/malicious client robustness.
- No formal theoretical convergence guarantee (unlike Paper 2).
- Uses **only 3 metrics**; doesn't formally integrate resource availability, dropout risk, or data heterogeneity into the score itself, despite discussing them conceptually.
- Weight adjustment (δ fixed, directionally binary) is heuristic, not derived from optimization theory.

---

## Paper 2: SubTrunc & UnionFL — Submodular Maximization for Equitable Client Selection
**Castillo, Kaya, Ye, Hashemi (arXiv 2024/2025)**

### Core Idea
Frame client selection as **submodular function maximization** (facility-location style) under a cardinality constraint, then add a **fairness-promoting regularization term** to the objective. Two variants proposed: **SubTrunc** (loss-based truncation) and **UnionFL** (selection-history-based).

### Base Objective (from DivFL, prior work)
Select subset S of clients whose aggregated gradient best approximates the full-client gradient:
```
max_S  G(S)   s.t. |S| ≤ κ
```
where G(S) is a **monotone submodular facility-location function** built from gradient similarity between clients (i.e., a "representative subset" objective, not accuracy/loss directly).

### SubTrunc: Fairness via Loss-Based Truncation
Adds a **truncated submodular regularizer**:
```
H(S) = λ · min(b, F(S))       where F(S) = Σ_{i∈S} φ(f_i(w))
```
- f_i(w) = client i's local loss
- φ = monotone nondecreasing function (e.g., ln(1+x) or identity) — controls how loss differences are attenuated/enhanced
- λ = weight on fairness term (trade-off knob between performance and fairness)
- b = truncation cap (large b → favors worst-performing clients; small b → objective becomes independent of local performance, i.e., purely representativeness-driven)

Final objective:
```
max_S  W(S) = G(S) + H(S)    s.t. |S| ≤ κ
```
Proven to remain **monotone submodular**, so greedy/stochastic-greedy algorithms give near-optimal solutions with guarantees.

### UnionFL: Fairness via Selection-History Regularization
Penalizes selecting clients that were **already selected recently** (within a look-back window u_t):
```
max_{S_t} f_t(S_t) − μ · g_t(S_t)    s.t. |S_t| ≤ K_t
```
- g_t(S_t) = |(⋃_{i∈u_t} S_i) ∩ S_t| — counts overlap with past window's selections (proven **supermodular**, so its negative is submodular, keeping the combined objective submodular)
- μ = regularization strength; higher μ → more diversification/rotation of clients over time
- u_t = look-back window (how many past rounds' selections to penalize overlap with)

### Selection Mechanism
- Stochastic **greedy algorithm**: at each step, sample a random candidate subset R ⊂ N, add the element with highest **marginal gain** Δ(e|S_k), repeat until κ clients chosen.
- Applies to either the SubTrunc or UnionFL objective.

### Theoretical Guarantees
- Proves convergence to an ε-accurate first-order stationary point under standard nonconvex FL assumptions, **without** requiring the restrictive Bounded Client Dissimilarity (BCD) assumption used in prior work (DivFL), and without assuming strong convexity.
- Convergence rate: K = O(1/ε²) communication rounds.

### Evaluation Metric: Client Dissimilarity
Instead of just accuracy/loss, they explicitly measure **fairness** via "client dissimilarity" — variance of the final model's per-client test performance. Lower = more equitable.

### Findings
- SubTrunc achieves the **lowest client dissimilarity** on non-IID MNIST/CIFAR-10 vs. DivFL, UnionFL, Random, Power-of-Choice — while maintaining comparable/better accuracy.
- Increasing λ (SubTrunc) or window size (UnionFL) improves fairness with only marginal effect on accuracy — reveals a **tunable trade-off**, not just a fixed heuristic.

### Strengths
- Formal theoretical grounding (submodularity + convergence proofs).
- Directly optimizes for **fairness/equity**, not just average accuracy.
- Selection cost is principled (greedy approximation ratio for submodular maximization), not ad hoc top-k.

### Limitations / Gaps
- Relies on **gradient similarity** computation (facility location) which is more computationally/communication expensive than Paper 1's simple accuracy/loss/time scalars — clients or server need gradient info, not just scalar metrics.
- No explicit resource-awareness (execution time, bandwidth, dropout) — purely performance/fairness-driven.
- Tuning λ, b, μ, u_t adds hyperparameter complexity.

---

## Paper 3: FedEntOpt — Entropy-Based Client Selection
**Lutz, Steidl, Müller, Samek (arXiv 2024/2025)**

### Core Idea
Select clients to **maximize the Shannon entropy of the aggregated label distribution** of the selected subset — directly targeting label-skew (non-IID) problems, rather than accuracy/loss/gradient metrics.

### Key Assumption
Global label marginal distribution is uniform; the goal is for the *selected subset's* combined label distribution to approximate this uniform global distribution.

### Setup
- Each client k precomputes and sends a **label count vector** l^(k) ∈ ℕ^C (count of each class c in its local data) to the server **once, before training** (minimal communication overhead vs. sending model gradients).

### Scoring/Selection Formula
Greedy entropy maximization. Starting from an aggregated label vector L (initialized with a randomly chosen first client), iteratively add the client that maximizes:
```
argmax_{m ∈ A\{k0}}  H( (L + l^(m)) / ||L + l^(m)||_1 )
```
where H(p) = −Σ p_c log₂(p_c) is Shannon entropy of the normalized combined label distribution.

### Selection Mechanism (Algorithm)
1. Sample first client k0 uniformly at random.
2. Repeat until M clients selected: pick the client (from those not in an "exclusion buffer") whose addition to the running aggregated label vector L maximizes entropy.
3. Maintain a **FIFO buffer B** of size Q — recently selected clients are excluded from re-selection (prevents deterministic repeated subsets), similar in spirit to UnionFL's window mechanism in Paper 2, but simpler (a hard exclusion buffer rather than a soft supermodular penalty).
4. Constraint: 0 < Q ≤ K − M (must always leave enough eligible clients).

### Privacy Extension
Applies **differential privacy** (Laplace mechanism) to the uploaded label count vectors:
```
A(l^(k)) = l^(k) + (Z_1,...,Z_C),  Z_i ~ Laplace(1/ε)
```
Shown empirically to barely affect (sometimes even improve) accuracy at ε = 0.5.

### Findings
- Outperforms FedAvg, FedProx, FedNova, SCAFFOLD, FedRS, FedLC, FedConcat, and a KL-divergence-based method by up to **6% accuracy** in standard settings, and **>30%** under low participation / client dropout.
- Robust to differential privacy noise.
- Robust under client dropout and straggling — because entropy-based selection doesn't rely on trained-model performance metrics that would be invalidated by a dropped/incomplete client.
- Buffer size Q trades off diversity (higher entropy of *who gets selected over time*, measured via normalized selection entropy H_norm) vs. accuracy — optimal Q depends on partition type (extreme skew Dir(0.1) prefers smaller buffer/less forced diversity; balanced quantity-skew C=2 prefers larger buffer).

### Strengths
- Extremely **cheap communication** (only C scalars per client, once — not gradients or repeated metrics).
- Directly and provably targets the root cause of non-IID degradation: label distribution mismatch.
- Strong empirical robustness to dropout/straggling because it doesn't depend on *live* performance metrics per round.
- Compatible/composable with other heterogeneity-handling methods (FedProx, SCAFFOLD, etc.) — used as a selection layer, giving further gains when combined.

### Limitations / Gaps
- **Ignores model performance entirely** (no accuracy/loss/execution-time signal at all) — could select clients with poor compute or slow execution.
- Assumes global label distribution is uniform; degrades if this assumption doesn't hold.
- Requires label-count sharing (mild privacy leakage risk, mitigated but not eliminated by DP).
- No fairness-of-participation guarantee beyond the FIFO buffer heuristic — no formal fairness bound like Paper 2's submodular theory.
- No resource-awareness (execution time, energy, communication cost of training) — purely data-distribution-driven.

---

## Cross-Paper Comparison Table

| Aspect | Paper 1 (Scoring: Acc/Time/Loss) | Paper 2 (SubTrunc/UnionFL) | Paper 3 (FedEntOpt) |
|---|---|---|---|
| **Primary signal used** | Accuracy, execution time, loss | Gradient similarity + local loss | Label distribution / entropy |
| **Selection method** | Top-k by weighted linear score | Greedy submodular maximization | Greedy entropy maximization |
| **Adaptivity** | Weights (α,β,γ) adjusted via moving-average trend each round | λ, b, μ, window size are static hyperparameters (tunable, not adaptive per round) | Static algorithm; buffer size Q is the main tunable knob |
| **Fairness mechanism** | None explicit (implicit via not fully excluding low scorers—only loosely) | Explicit: truncated loss regularizer (SubTrunc) or selection-history penalty (UnionFL) | Implicit: FIFO exclusion buffer prevents repeated same-subset selection |
| **Theoretical guarantees** | None | Yes — submodularity + O(1/ε²) convergence bound, no BCD assumption needed | None (empirical only) |
| **Communication overhead** | Requires accuracy/loss/time metrics reported each round | Requires gradient info each round (heavier) | One-time label count vector (very cheap) |
| **Robustness to dropout/stragglers** | Not directly addressed | Not directly addressed | Explicitly tested — strong robustness |
| **Privacy consideration** | Not addressed | Not addressed | Differential privacy (Laplace mechanism) applied and tested |
| **Resource-awareness (CPU/bandwidth/energy)** | Execution time only | None | None |
| **Data heterogeneity targeted** | Indirectly (via accuracy/loss reflecting poor fit) | Indirectly (via gradient dissimilarity + loss) | Directly (label skew is the explicit target) |
| **Evaluation domain** | Healthcare (diabetes, tabular, Random Forest) | Vision (MNIST, CIFAR-10, LeNet) | Vision + medical imaging (CIFAR, CINIC, PathMNIST, TissueMNIST, Brain-Tumor MRI) |

---

## Implications for Designing a Better Scoring System

Based on the gaps above, a stronger scoring-based client selection method could combine ideas across all three papers:

1. **From Paper 1**: Keep a lightweight, interpretable weighted-score formula (cheap to compute, no gradient sharing needed) and its *dynamic weight adaptation* idea — but make the adaptation rule more principled (e.g., gradient-based or bandit-based weight updates rather than a fixed ± δ).
2. **From Paper 2**: Add a **formal fairness regularization term** (truncated-loss or selection-history-penalty style) to the score so it isn't a pure top-k on raw performance — and ideally prove the resulting score/selection function retains desirable structure (e.g., submodularity) so greedy selection has guarantees.
3. **From Paper 3**: Incorporate a **data-distribution/entropy-awareness term** into the score (not just accuracy/loss) so that label-skew is directly addressed, and consider **cheap one-time metadata sharing** (label counts, or similar lightweight statistics) instead of relying purely on round-by-round performance metrics — this also improves robustness to dropout/stragglers, since entropy-based signals don't depend on a client successfully finishing local training.
4. **Add resource-awareness + dropout-risk modeling** (Paper 1 gestures at this conceptually but never formalizes it) — e.g., predicted dropout probability and bandwidth as additional score terms.
5. **Add privacy consideration** (Paper 3's DP mechanism) for whichever metrics are shared with the server, since accuracy/loss/label-counts are all potentially privacy-sensitive.
6. **Provide theoretical convergence/fairness guarantees** (Paper 2's approach) for the final proposed scoring formula, which none of Papers 1 or 3 currently offer for their respective methods.

A composite score of the form:
```
Score_i = α·accuracy_i − β·norm_time_i − γ·norm_loss_i + δ·entropy_contribution_i + λ·fairness_term_i − ρ·dropout_risk_i
```
with theoretically justified (e.g., submodular-preserving) fairness/entropy terms and a principled adaptive-weight scheme, would directly address the identified gaps in all three papers.
