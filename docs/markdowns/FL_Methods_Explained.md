# Federated Learning Methods: FedEntOpt, FedPrism, and FedProx

Federated learning (FL) trains a shared model across many clients holding private, non-IID data. The three methods below tackle heterogeneity from different angles: **FedEntOpt** fixes *which* clients are selected, **FedPrism** fixes *how personalization is structured*, and **FedProx** fixes *how local updates are regularized*. Each section explains the mechanism, the mathematical intuition, its advantages, and its limitations.

---

## 1. FedEntOpt — Entropy-Based Client Selection

### Mechanism

Standard FedAvg selects a random subset of clients each round and aggregates their updates. Under label skew — where each client's local label marginal $P_{Y^{(k)}}$ differs from the global marginal $P_Y$ — a random subset can produce an aggregated label distribution far from uniform, biasing the global model.

FedEntOpt reframes this as a **combinatorial selection problem**. Before training, each client $k$ sends a label-count vector $l^{(k)} \in \mathbb{N}^C$ (counts per class, not raw data). The server maintains a running aggregate $L \in \mathbb{R}^C$ and greedily grows the selected set $S$:

$$
m^* = \arg\max_{m \in A \setminus B} \; H\!\left(\frac{L + l^{(m)}}{\lVert L + l^{(m)} \rVert_1}\right), \qquad H(p) = -\sum_{c=1}^{C} p_c \log_2 p_c
$$

where $H$ is Shannon entropy and $B$ is a FIFO buffer excluding recently-picked clients (to prevent the same "well-balanced" cohort from being reused every round). At each step, the client added is the one that pushes the *combined* label distribution of the selected set closest to uniform — the entropy-maximizing distribution.

A useful algebraic fact: the aggregated distribution is exactly a data-size-weighted mixture of local empirical distributions,
$$
\frac{L_c}{\lVert L \rVert_1} = \sum_{k \in S} \left(\frac{n_k}{\sum_{j \in S} n_j}\right) \tilde{P}^{(k)}_Y(c),
$$
so maximizing entropy over $S$ is equivalent to greedily approximating the uniform global marginal via a weighted combination — no separate reweighting step is needed at aggregation time.

### Advantages

- **Directly targets the failure mode of label skew** by optimizing the class coverage of the *selected cohort*, not just each client's local loss — up to 6% higher accuracy in standard settings, and >30% gains under low participation (4–9%) or client dropout, where random selection is most likely to miss classes entirely.
- **Negligible communication overhead**: label vectors cost $4CK$ bytes once, versus $4p$ bytes per client per round for model parameters ($p$ = number of parameters) — typically 3–4 orders of magnitude cheaper.
- **Orthogonal to client-side heterogeneity methods** (FedProx, SCAFFOLD, FedNova, FedRS, FedLC): since it only changes *who* is sampled, it composes with these methods and improves their accuracy further (>40% in some Dir(0.1) settings).
- **Differential privacy is nearly free**: adding Laplace noise ($\epsilon = 0.5$) to label counts before upload leaves accuracy essentially unchanged, because the selection only needs approximate class proportions, not exact ones.
- **Deterministic and lightweight**: no extra local computation, soft-label inference, or model-based similarity metric is required — just integer counts and a greedy argmax.

### Limitations

- **Relies on the fidelity of self-reported label counts.** A client can lie about its counts (report a class it doesn't have, or inflate rare classes) to bias selection; the paper notes this is a real threat model without a dedicated defense beyond limiting any one client's influence via participation caps and buffer size.
- **Degrades under extreme sparsity + very low participation**: if most clients are near single-class and only a handful can be selected per round, entropy of the aggregate can plateau below $\log_2(C-1)$, leaving some classes systematically underrepresented.
- **The buffer size $Q$ is a sensitive hyperparameter** whose optimum differs by skew type (empirically $Q \approx 50\%$ of clients for Dirichlet-based skew vs. $Q \approx 70\%$ for quantity-based skew) — it is not skew-agnostic.
- **Selection-only scope**: it says nothing about how a client trains locally, so it still needs to be paired with a local-optimization method to address systems heterogeneity (partial work, stragglers) directly — although this composability is also framed as a strength.

---

## 2. FedPrism — Adaptive Personalized FL via Global–Cluster–Private Decomposition

### Mechanism

FedPrism targets **personalization** rather than selection: instead of one global model $w_G$, each client $i$'s model is a weighted sum of three components,

$$
w_i = \alpha_i w_G + \beta \sum_{k=1}^{K} \pi_{i,k} C_k + \gamma_i P_i,
$$

where $w_G$ is the shared global backbone (aggregated over all participants), $C_k$ are $K$ server-maintained cluster models, $\pi_{i,k}$ is client $i$'s soft assignment weight to cluster $k$, and $P_i$ is a private component trained only locally and never transmitted.

**Dynamic soft clustering.** Rather than a one-shot or hard cluster assignment (as in IFCA or FedClust), FedPrism re-clusters every few rounds using each client's classifier-layer weights as a "prototype" $h_i$. The server runs K-means on collected prototypes to get centroids $\mu_k$, and each client's cluster weight is a softmax over cosine similarity:
$$
w_{i,k} \propto \exp\!\big(\text{sim}(h_i, \mu_k)/\tau\big).
$$
This lets a client belong to multiple clusters simultaneously and drift between clusters as its local distribution (or the global data landscape) evolves — addressing the rigidity of hard clustering methods.

**Dual-stream inference.** In parallel, each client also keeps a fully independent **local expert** $L_i$, trained only on local data. At inference time on input $x$, the routing weight is the expert's own confidence:
$$
\lambda(x) = \max \; \text{Softmax}\!\left(\frac{\text{Expert}(x)}{T}\right), \qquad y_{\text{pred}} = \lambda(x)\cdot \text{Expert}(x) + (1-\lambda(x))\cdot \text{Backbone}(x).
$$
High confidence routes the prediction toward the specialist; low confidence falls back to the generalizable backbone.

The mixing coefficient $\alpha_i$ is not fixed — it is updated adaptively based on whether the collaborative model $W_\text{Backbone}$ is closer to the client's assigned cluster or to the global model, via a signed update $\alpha_i \leftarrow \text{Clip}(\alpha_i + \eta_\alpha \Delta, [0,1])$.

### Advantages

- **Handles mixed/hybrid client identities**: hard-clustering methods (IFCA, FedClust, CFL) force each client into exactly one group, which is a poor fit when a client's distribution overlaps several latent groups. Soft assignment via $\pi_{i,k}$ avoids this rigidity.
- **Strong empirical personalization gains under extreme heterogeneity**: on CIFAR-100 at Dirichlet $\alpha=0.1$, local accuracy of 39.91% versus 13.48% (FedAvg) and 8.00% (FedClust) — roughly triple the strongest global baseline.
- **Mitigates negative transfer**: on pathological (disjoint-class) partitions, FedPrism nearly matches pure local training (94.02% vs. 94.01% on SVHN local accuracy) while FedAvg collapses to 79.28%, showing the gating mechanism successfully filters out harmful cross-client interference.
- **Ablations isolate each component's role**: the global backbone drives global accuracy, the private component stabilizes local accuracy (even alone it reaches >83% locally), and the dual-stream expert is what prevents local accuracy from collapsing to global-only levels (12% → 82%+ once expert weight is nonzero).
- **No manual cluster-count tuning at inference**: because assignment is similarity-based and continuously updated, it adapts to concept drift without re-running a full clustering pipeline from scratch.

### Limitations

- **Added system complexity**: three model components per client plus a separate local expert roughly doubles per-client storage/compute relative to FedAvg, and requires periodic server-side K-means over prototypes — more moving parts than a single global model.
- **Global accuracy can be highly sensitive to $\alpha$**: in the pathological CIFAR-10 ablation, global accuracy swings from ~11% (α ≤ 0.5) to 38.47% (α = 0.9) while local accuracy stays flat — meaning the global-generalization behavior is fragile to this one hyperparameter and must be tuned per setting.
- **K (number of clusters) is fixed and pre-specified**, unlike CFL, which grows clusters as needed — a poor choice of $K$ could misrepresent the true number of latent groups.
- **Local expert cannot generalize by design**: it is trained purely on local data, so on genuinely novel inputs (outside a client's own distribution) the system depends entirely on correctly detecting low expert-confidence to fall back to the backbone; miscalibrated confidence would misroute predictions.
- **Prototype sharing leaks some information**: transmitting classifier-layer weights every clustering round, while not raw data, still exposes model-level statistics that could in principle be used to infer distributional properties — a privacy consideration the paper does not formally analyze (unlike FedEntOpt's explicit DP treatment).

---

## 3. FedProx — Proximal Regularization for Systems + Statistical Heterogeneity

### Mechanism

FedProx generalizes FedAvg by modifying the **local subproblem** each client solves, rather than changing which clients are picked or how models are structured. Instead of minimizing the local objective $F_k(w)$ directly, each client minimizes

$$
h_k(w; w^t) = F_k(w) + \frac{\mu}{2}\lVert w - w^t \rVert^2,
$$

i.e., it adds a quadratic **proximal term** anchoring the local update to the last global model $w^t$. This does two things simultaneously:

1. **Tolerates partial work.** Because devices have heterogeneous compute/network resources, forcing a uniform number of local epochs $E$ is unrealistic. FedProx allows each client $k$ at round $t$ to return a $\gamma^t_k$-inexact solution (Definition 2), i.e., the client can do as much or as little local optimization as its resources allow, and the server still aggregates the partial result instead of discarding stragglers as FedAvg does.
2. **Bounds client drift.** The proximal term keeps $w_k^{t+1}$ close to $w^t$, so no single client's local optimum (which may be very different from the global optimum under statistical heterogeneity) is allowed to pull the aggregate too far.

### Mathematical basis

The analysis relies on a **bounded dissimilarity** assumption: for $B(w) = \sqrt{\mathbb{E}_k[\lVert \nabla F_k(w)\rVert^2]} / \lVert \nabla f(w)\rVert$, assume $B(w) \le B$ for all $w$ outside a small-gradient region. $B=1$ recovers the IID case; $B>1$ quantifies statistical heterogeneity. Under this assumption and $\mu$-strong convexity of $h_k$ (guaranteed once $\mu > L^-$, the local curvature lower bound), one round of FedProx gives a guaranteed expected decrease:

$$
\mathbb{E}_{S_t}\!\left[f(w^{t+1})\right] \le f(w^t) - \rho \lVert \nabla f(w^t) \rVert^2, \qquad \rho > 0 \text{ for appropriate } \mu, K, \gamma.
$$

This is the first convergence guarantee of its kind for a method combining local updating, partial participation, *and* non-IID data. Setting $\mu = 0$ with SGD as the local solver exactly recovers FedAvg — so FedProx is a strict generalization, not a separate algorithm family.

### Advantages

- **Formal convergence guarantee under both heterogeneity types simultaneously** (statistical *and* systems) — something FedAvg lacks; prior analyses of local-updating SGD variants assumed either IID data or full participation, not both relaxed at once.
- **Empirically stabilizes and improves accuracy in high-heterogeneity regimes**: +22% absolute test accuracy on average over FedAvg when 90% of devices are stragglers.
- **No architectural change required**: FedProx is a drop-in modification to the local loss function; existing FedAvg infrastructure (TensorFlow Federated, LEAF, etc.) needs minimal changes.
- **A tunable knob for drift**: the proximal coefficient $\mu$ can be adjusted (even adaptively — increase when loss rises, decrease when loss falls for several consecutive rounds) to interpolate between aggressive local optimization (small $\mu$) and conservative, global-model-anchored updates (large $\mu$), without needing to hand-tune the number of local epochs $E$ per device.
- **Degrades gracefully to FedAvg when heterogeneity is low**: on IID synthetic data, FedProx with $\mu>0$ performs comparably (slightly slower initially), so there's little downside to using it as a default.

### Limitations

- **$\mu$ requires tuning per dataset**, typically searched over a small grid ($\{0.001, 0.01, 0.1, 1\}$); the best value varied substantially across datasets in the paper (from 0.001 on Shakespeare to 1 on Synthetic/MNIST/FEMNIST), so there is no universal default.
- **Does not address *which* clients are chosen** — it accepts whatever cohort is sampled (uniformly at random in the paper's experiments) and only regularizes what happens locally; it doesn't correct for a badly skewed selected subset the way FedEntOpt does.
- **The convergence theory has restrictive sufficient conditions**: e.g., Corollary 7 requires $B \le 0.5\sqrt{K}$, meaning the guarantee only formally holds when statistical heterogeneity is bounded relative to the number of participating devices $K$ — highly skewed settings with small $K$ fall outside the theorem's guarantees even though the method may still work empirically.
- **A large $\mu$ can slow convergence unnecessarily** on near-IID data by over-constraining updates to stay near $w^t$, trading off some optimization speed for stability that may not be needed.
- **No personalization**: like FedAvg, FedProx still converges to a single global model; it does not offer per-client specialization the way FedPrism does, so under extreme label skew, individual clients may still see suboptimal local performance even if the global objective converges well.

---

## Summary: What Each Method Actually Changes

| Method | Changes... | Core mechanism | Not addressed |
|---|---|---|---|
| **FedEntOpt** | Client *selection* | Greedy entropy maximization over aggregated label counts | Local training heterogeneity, personalization |
| **FedPrism** | Model *structure* per client | Global+Cluster+Private decomposition with soft dynamic clustering and confidence-gated dual-stream inference | Systems heterogeneity (stragglers), formal convergence guarantees |
| **FedProx** | Local *optimization objective* | Proximal term anchoring local updates to the global model, tolerating inexact/partial solutions | Client selection, personalization |

These three are complementary rather than competing: FedEntOpt's authors explicitly show it composing with FedProx and other client-side methods to yield further gains, since selection and local optimization operate on different parts of the pipeline.
