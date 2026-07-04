# PQC_CSS: PQC-aware Client-Selection Strategy

**A client-selection strategy for Federated Learning that factors in the heterogeneous
computational and communication costs of Post-Quantum Cryptography (PQC).**

## Idea

Instead of selecting clients purely at random (FedAvg default) or by data size alone,
each client reports, alongside training metrics:

- **Learning utility** — local data size, local loss
- **Device/network profile** — simulated CPU score, bandwidth, dropout probability, trust
- **PQC cost** — real measured ML-KEM-768 (key exchange) keygen/encaps/decaps time,
  ML-DSA-65 (signature) sign/verify time, and PQC bytes transmitted

These are combined into a **PQC Cost Index (PCI)** and a composite selection score:

```
score = 0.35*utility + 0.20*trust + 0.15*reliability
        - 0.15*PCI - 0.10*comm_cost - 0.05*dropout_risk
```

The server keeps the top-K scoring clients each round instead of sampling randomly.

## Project layout

```
PQC_CSS/
├── README.md
└── css/
    ├── LICENSE
    ├── pyproject.toml
    └── c_select/
        ├── __init__.py
        ├── task.py        # model, data loading, train/test (unchanged)
        ├── pqc.py          # ML-KEM-768 / ML-DSA-65 cost measurement
        ├── selection.py    # PCI + PQC-aware score + baseline scorers
        ├── strategy.py     # ScoredSelectionStrategy (FedAvg + pluggable scoring)
        ├── client_app.py   # reports PQC + device metrics each round
        └── server_app.py   # wires the strategy, picks scorer from config
```

## Install (you're already inside the `PQC_CSS` git repo)

```bash
cd css
pip install -e .
```

This pulls in `pqcrypto` (PQClean bindings providing standardized ML-KEM /
ML-DSA — the FIPS 203/204 names for Kyber/Dilithium) alongside Flower and
PyTorch. No separate clone of liboqs is needed — `pqcrypto` ships prebuilt
wheels.

## Run

```bash
flwr run .
```

Switch the selection policy (and compare against baselines) by editing
`selection-strategy` in `pyproject.toml`'s `[tool.flwr.app.config]`:

`pqc_aware` (ours) | `random` (FedAvg default) | `data_size` | `loss_based` |
`resource_aware` | `trust_based`

Round 1 always samples randomly (no client has reported metrics yet); from
round 2 onward the chosen scorer ranks clients using their last reported
metrics.

## Notes

- `pqc.py` measures *real* cryptographic operations (not mocked numbers) using
  the NIST-standardized ML-KEM-768 / ML-DSA-65 parameter sets, so the PCI
  reflects actual algorithmic cost, not guesses.
- `resource_aware`'s CPU/bandwidth/dropout/trust values are simulated
  per-partition (seeded, so stable across rounds) since Flower's simulation
  engine doesn't expose real device telemetry — swap
  `_simulated_device_profile` in `client_app.py` for real sensors when
  moving off simulation.

