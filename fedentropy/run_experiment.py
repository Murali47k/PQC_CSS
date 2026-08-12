"""
run_experiment.py
------------------
Runs FedAvg and FedEntOpt back-to-back on CIFAR-10 with ResNet-18 under a
label-skew partition, using Flower's simulation engine and flwr-datasets
for data loading (CIFAR-10 is pulled from HuggingFace on first use and
cached in ~/.cache/huggingface -- no manual download, no ./data folder).
Centralized (server-side, held-out test set) accuracy/loss per round is
written to `results/<strategy>.csv`.

Example
-------
Quick smoke run (small, CPU-friendly, a few minutes):

    python run_experiment.py --num-clients 20 --clients-per-round 5 \
        --rounds 10 --local-epochs 1 --partition dirichlet --alpha 0.1

Paper-scale run (100 clients, 500 rounds, needs a GPU and hours):

    python run_experiment.py --num-clients 100 --clients-per-round 10 \
        --rounds 500 --local-epochs 5 --partition dirichlet --alpha 0.1
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from client_app import build_client_app
from server_utils import make_evaluate_fn
from strategy import FedEntOptStrategy, weighted_average
from task import NUM_CLASSES, compute_all_label_counts


def parse_args():
    p = argparse.ArgumentParser(description="FedEntOpt vs FedAvg on CIFAR-10 / ResNet-18")
    p.add_argument("--num-clients", type=int, default=20, help="Total client pool size K")
    p.add_argument("--clients-per-round", type=int, default=5, help="M, clients sampled/round")
    p.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    p.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument(
        "--partition", choices=["dirichlet", "shard", "iid"], default="dirichlet",
        help="Label-skew simulation strategy (dirichlet = Dir(alpha), shard = quantity-based C=j)",
    )
    p.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration (Dir(alpha))")
    p.add_argument("--classes-per-client", type=int, default=2, help="j, for --partition shard")
    p.add_argument("--buffer-frac", type=float, default=0.5, help="FedEntOpt buffer size as frac of K")
    p.add_argument("--num-cpus", type=int, default=1, help="CPUs per simulated client")
    p.add_argument("--num-gpus", type=float, default=0.0, help="GPUs per simulated client (fraction)")
    p.add_argument("--out-dir", type=str, default="./results")
    p.add_argument(
        "--strategies", nargs="+", default=["fedavg", "fedentopt"],
        choices=["fedavg", "fedentopt"], help="Which strategies to run",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one(strategy_name: str, args, label_counts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Running {strategy_name.upper()} on device={device} ===")

    client_app = build_client_app(
        num_partitions=args.num_clients,
        local_epochs=args.local_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        partition=args.partition,
        alpha=args.alpha,
        classes_per_client=args.classes_per_client,
        seed=args.seed,
    )

    round_metrics = []  # filled in by evaluate_fn as a side effect

    # Built once so the same in-memory model + test DataLoader are reused
    # across every round's evaluation call (faster than rebuilding them).
    eval_fn = make_evaluate_fn(device, batch_size=128)

    def evaluate_fn(server_round, parameters, config):
        loss, metrics = eval_fn(server_round, parameters, config)
        round_metrics.append({"round": server_round, "loss": loss, "accuracy": metrics["accuracy"]})
        print(f"  [{strategy_name}] round {server_round:3d} | loss {loss:.4f} | acc {metrics['accuracy']*100:.2f}%")
        return loss, metrics

    common_kwargs = dict(
        fraction_fit=args.clients_per_round / args.num_clients,
        fraction_evaluate=0.0,  # we only use centralized evaluation
        min_fit_clients=args.clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=weighted_average,
    )

    if strategy_name == "fedavg":
        strategy = FedAvg(**common_kwargs)
    elif strategy_name == "fedentopt":
        strategy = FedEntOptStrategy(
            label_counts=label_counts,
            num_classes=NUM_CLASSES,
            clients_per_round=args.clients_per_round,
            buffer_frac=args.buffer_frac,
            selection_seed=args.seed,
            **common_kwargs,
        )
    else:
        raise ValueError(strategy_name)

    def server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=args.rounds))

    server_app = ServerApp(server_fn=server_fn)

    backend_config = {
        "client_resources": {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus}
    }

    t0 = time.time()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config=backend_config,
    )
    print(f"  {strategy_name} finished in {time.time() - t0:.1f}s")

    return round_metrics


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "loss", "accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Preparing {args.num_clients} CIFAR-10 partitions via flwr-datasets "
          f"('{args.partition}') -- downloaded once, cached in ~/.cache/huggingface ...")
    label_counts = compute_all_label_counts(
        args.num_clients,
        partition=args.partition,
        alpha=args.alpha,
        classes_per_client=args.classes_per_client,
        seed=args.seed,
    )
    sizes = [int(v.sum()) for v in label_counts.values()]
    print(f"  client dataset sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

    for strategy_name in args.strategies:
        rows = run_one(strategy_name, args, label_counts)
        save_csv(rows, os.path.join(args.out_dir, f"{strategy_name}.csv"))

    print("\nDone. Now run: python analyze_results.py --results-dir", args.out_dir)


if __name__ == "__main__":
    main()
