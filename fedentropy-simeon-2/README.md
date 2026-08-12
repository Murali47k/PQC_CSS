# FedEntropy on Flower (Proof of Concept)

A small, self-contained reimplementation of the **FedEntropy** idea from
*"FedEntropy: Efficient Federated Learning for Non-IID Scenarios Using
Maximum Entropy Judgment-based Device Selection"*, built on top of
[Flower](https://flower.ai) so all the federated learning "plumbing"
(client-server communication, scheduling, weighted aggregation) is
handled for you.

This is intentionally simple: one small CNN, one dataset, ~6 short files,
no CLI project scaffolding. The goal is to make the *idea* easy to read
and easy to extend, not to reproduce paper-level benchmarks.

## The idea, in one paragraph

In federated learning, each client trains a local model on its own
private data and sends the weights to a server, which averages them into
a global model. When clients' data is **non-IID** (e.g. each device only
has photos of 1-2 classes), some clients' local models become skewed
toward their own class(es), and blindly averaging every client's weights
drags the global model off balance. FedEntropy's fix: after training,
each client also reports a **soft label** — the average predicted
class-probability vector over its own data. The server combines these
soft labels (never the raw data) and greedily drops whichever clients'
soft labels make the *combined* distribution more skewed, keeping only
the subset whose combined soft labels are most balanced (highest
entropy) — then aggregates only the surviving clients' weights.

## Project structure

```
model.py             A small CNN (2 conv layers + 2 FC layers)
data.py               Dataset loading + non-IID Dirichlet partitioning
common.py              Shared train/evaluate/soft-label helper functions
client.py               The Flower client (local training + soft label)
entropy_utils.py         The core FedEntropy math, isolated & unit-testable
entropy_strategy.py       A Flower Strategy that filters clients using entropy_utils
main.py                    Wires everything together and runs the simulation
```

Reading order if you're new to this codebase: `entropy_utils.py` (the
actual idea) → `client.py` → `entropy_strategy.py` → `main.py`.

## How it maps onto Flower

| Concept | Where |
|---|---|
| A federated learning participant | `client.py` — a `flwr.client.NumPyClient` |
| "Which clients do we trust this round?" | `entropy_strategy.py` — a `flwr.server.strategy.FedAvg` subclass |
| Turning trusted clients' weights into one model | Inherited, unmodified, from `FedAvg` |
| Running many simulated clients on one machine | `main.py` — `flwr.simulation.run_simulation` |

The only FedEntropy-specific code is: (a) clients compute and report a
soft label in `client.py`, and (b) the server filters by entropy before
aggregating in `entropy_strategy.py`. Everything else — sampling clients
each round, retrying failures, weighted-averaging parameters, running
many virtual clients in parallel — is standard Flower behavior we get
for free.

## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
python main.py
```

You'll see output like:

```
Loading dataset and building a non-IID partition across clients...
  client 0: 879 samples
  ...
[Round 1] FedEntropy filter: kept 4/4 clients (0 dropped as likely to bias the aggregation).
[Round 1] global model -> loss=2.0643, accuracy=0.1990
...
[Round 6] FedEntropy filter: kept 4/4 clients (0 dropped as likely to bias the aggregation).
[Round 6] global model -> loss=0.4014, accuracy=0.9130
```

Useful flags (see `python main.py --help`):

```bash
# More clients, more rounds, more skew (lower alpha = more non-IID)
python main.py --num-clients 20 --num-rounds 10 --alpha 0.1

# Less skewed data (closer to IID) - the entropy filter should drop fewer clients
python main.py --alpha 5.0
```

## About the dataset

By default the code tries to download real **MNIST** via `torchvision`.
If that fails (no internet access, e.g. in a sandboxed environment), it
automatically falls back to a small synthetic dataset (`SyntheticDigits`
in `data.py`) with the same shape (28x28 grayscale, 10 classes), so the
whole pipeline can still be run and inspected fully offline. When real
MNIST is available it's used automatically — no code changes needed.

## Extending this

- **Swap the model**: edit `model.py`. Nothing else needs to change, as
  long as the model still returns raw logits over `num_classes`.
- **Swap the dataset**: edit `load_datasets()` in `data.py` to return any
  two PyTorch `Dataset`s with a `.targets` attribute (CIFAR-10, your own
  data, ...).
- **Try a different non-IID split**: `data.py`'s `dirichlet_partition`
  is the only place partitioning happens; the shard-based ("each client
  gets 1-2 classes") split from the original codebase would be a
  reasonable second option to add there.
- **Add client-side (not just centralized) evaluation**: implement an
  `evaluate()` method on `FedEntropyClient` in `client.py` and set
  `fraction_evaluate > 0` in `main.py`.
- **Add the paper's adaptive client *sampling*** (this POC only
  implements the entropy-based *filtering* step, using Flower's default
  uniform-random client sampling): override `configure_fit` in
  `entropy_strategy.py` to sample proportionally to a per-client score
  that you update round over round, similar to `value` in the original
  `FedEntropy` training loop.

## A note on Flower's API

This project uses `flwr.simulation.run_simulation` with a plain
`ClientApp`/`ServerApp`, callable directly via `python main.py`. Recent
versions of Flower recommend the `flwr run` CLI workflow with a
`pyproject.toml`-based app instead; that path is more "production
shaped" but adds project scaffolding that isn't necessary for a small
POC. Everything here (`FedEntropyClient`, `FedEntropyStrategy`,
`entropy_utils.py`) would carry over unchanged if you later migrate.
