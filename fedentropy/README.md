# FedEntOpt vs. FedAvg on CIFAR-10 (ResNet-18) — Flower implementation

This is a runnable [Flower](https://flower.ai) implementation of **FedEntOpt**
("Optimizing Federated Learning by Entropy-Based Client Selection", Lutz et
al.), benchmarked against plain **FedAvg**, on CIFAR-10 with a ResNet-18
under label skew.

Data loading uses **flwr-datasets**, exactly like a standard `flwr new`
scaffold: CIFAR-10 (`uoft-cs/cifar10` on HuggingFace) is fetched once and
cached in `~/.cache/huggingface` on first run. There's no manual download
step and no `./data` folder to manage — `FederatedDataset` handles both
fetching and partitioning.

FedEntOpt replaces FedAvg's uniform-random client sampling with a **greedy,
entropy-maximizing selection** (Algorithm 1 in the paper): each client
uploads its label-count vector once, and each round the server greedily adds
the client whose data pushes the *aggregated* label distribution of the
selected subset closest to uniform (max Shannon entropy), using a FIFO
buffer to stop the same clients being picked every round. Model aggregation
itself is still standard size-weighted FedAvg — only *which clients train*
changes.

## Files

| File | Purpose |
|---|---|
| `task.py` | ResNet-18 (CIFAR-sized stem), Dirichlet & shard label-skew partitioning, train/test loops |
| `client_app.py` | Flower `ClientApp` / `NumPyClient` — local SGD training, reports its partition id via `get_properties` |
| `strategy.py` | `FedEntOptStrategy` (Algorithm 1) — a Flower `Strategy` overriding `configure_fit` |
| `server_utils.py` | Centralized (server-side) evaluation function on the CIFAR-10 test set |
| `run_experiment.py` | Orchestrates a full run for both `fedavg` and `fedentopt`, saves per-round metrics to CSV |
| `analyze_results.py` | Prints a summary table + saves a comparison plot from the CSVs |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This needs `flwr[simulation]` (which pulls in Ray, used to run many
simulated clients as parallel actors), `flwr-datasets[vision]`, PyTorch, and
torchvision. A GPU is optional but strongly recommended for anything beyond
a small smoke test — `--num-gpus` lets you assign a fraction of a GPU to
every simulated client.

CIFAR-10 is fetched automatically via `flwr_datasets.FederatedDataset` /
HuggingFace `datasets` on first run (cached under `~/.cache/huggingface`,
~170MB) — you don't need to point it at a local folder.

## Running

**Quick smoke test** (CPU-friendly, a few minutes, just to see the pipeline work):

```bash
python run_experiment.py \
  --num-clients 20 --clients-per-round 5 \
  --rounds 15 --local-epochs 1 \
  --partition dirichlet --alpha 0.1 \
  --out-dir results_smoke
```

**Paper-scale run** (100 clients, 10% participation, 500 rounds, GPU
recommended — this reproduces Table I's `Dir(0.1)` / LeNet-scale setting but
with ResNet-18 as requested; expect several hours on a single GPU):

```bash
python run_experiment.py \
  --num-clients 100 --clients-per-round 10 \
  --rounds 500 --local-epochs 5 \
  --batch-size 64 --lr 0.01 --weight-decay 0.0005 \
  --partition dirichlet --alpha 0.1 \
  --num-cpus 1 --num-gpus 0.1 \
  --out-dir results
```

**Quantity-based skew (the paper's `C = 2` setting)**, where every client
only ever holds 2 of the 10 CIFAR-10 classes:

```bash
python run_experiment.py \
  --num-clients 100 --clients-per-round 10 \
  --rounds 500 --local-epochs 5 \
  --partition shard --classes-per-client 2 \
  --num-gpus 0.1 --out-dir results_c2
```

Useful flags:

- `--strategies fedavg fedentopt` — run one or both (default: both, back-to-back).
- `--buffer-frac` — FedEntOpt's FIFO buffer size `Q` as a fraction of `K`
  (paper finds ~0.5 best for `Dir(0.1)`, ~0.7 best for `C=2`).
- `--num-gpus 0.0` forces CPU-only simulation.
- `--num-gpus 0.25` lets 4 simulated clients share one GPU concurrently.

Each run prints round-by-round centralized accuracy/loss and writes
`results/fedavg.csv` and `results/fedentopt.csv` with columns
`round, loss, accuracy`.

## Analyzing results

```bash
python analyze_results.py --results-dir results
```

This prints:
- final-round accuracy for each strategy,
- mean ± std accuracy over the last `--last-n` rounds (default 10 — matches
  how the paper reports numbers),
- FedEntOpt's absolute percentage-point improvement over FedAvg,

and saves `results/comparison.png`, a side-by-side accuracy-vs-round and
loss-vs-round plot for both strategies.

## Notes on the implementation

- **Label-count upload is simulated, not skipped.** In a real deployment
  each client sends its label-count vector to the server once, before
  training (this is what `FedEntOptStrategy` relies on). Since Flower's
  simulation backend assigns each simulated client an opaque random id, the
  strategy resolves `client id -> data partition id` once per client via
  Flower's built-in `get_properties` RPC (see `FlowerClient.get_properties`
  in `client_app.py`), then uses that partition's precomputed label counts.
  This is equivalent to, and exercises the same code path as, the one-time
  metadata exchange described in the paper — it does not give FedEntOpt
  access to anything a real deployment wouldn't have.
- **Evaluation is centralized**, on the full CIFAR-10 test set after every
  round, matching the paper's protocol and ensuring both strategies are
  compared on an identical metric.
- **FedAvg baseline** uses Flower's built-in `FedAvg` strategy with
  `fraction_fit = clients_per_round / num_clients`, i.e. uniform random
  sampling each round — the standard baseline used throughout the paper.
- **`--partition iid`** is also available if you want a sanity-check run
  with `IidPartitioner` (no label skew) before trying the skewed settings.
- To try other partitioning severities, adjust `--alpha` (smaller ⇒ more
  skewed; the paper's hardest setting is `Dir(0.1)`) or
  `--classes-per-client`.
- `run_simulation` (used here) is Flower's stable simulation API as of
  writing; if your installed Flower version has since moved fully to the
  `flwr run` CLI workflow, the core logic in `strategy.py` / `task.py` /
  `client_app.py` is unchanged — only the small orchestration block in
  `run_experiment.py` would need updating to match.
