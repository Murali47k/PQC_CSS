"""
main.py
=======
Entry point for the FedEntropy proof-of-concept, built on top of Flower.

Run with:
    python main.py
    python main.py --num-clients 20 --num-rounds 10 --alpha 0.1

Flower handles all the client-server orchestration: spinning up virtual
clients, sending them the global model, collecting their updates, and
retrying/failure handling. We only had to provide:
  1. A Flower Client that trains locally and reports a soft label
     (client.py)
  2. A Flower Strategy that filters clients based on those soft labels
     before aggregating (entropy_strategy.py)

Everything below is just wiring those pieces together and running a local
simulation with `num_clients` virtual clients on this one machine.
"""

import argparse
import warnings

from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from torch.utils.data import DataLoader

from client import FedEntropyClient
from common import evaluate as evaluate_model
from common import set_weights
from data import dirichlet_partition, load_datasets, make_client_subset
from entropy_strategy import FedEntropyStrategy
from model import SimpleCNN

warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLASSES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedEntropy proof-of-concept (built on Flower)")
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of simulated clients")
    parser.add_argument("--fraction-fit", type=float, default=0.5, help="Fraction of clients sampled each round")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of federated training rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per client per round")
    parser.add_argument("--batch-size", type=int, default=32, help="Local training batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Local SGD learning rate")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration (lower = more non-IID)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the data partition")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- 1. Data: load once, then split into non-IID shards --------------
    print("Loading dataset and building a non-IID partition across clients...")
    train_dataset, test_dataset = load_datasets()
    client_indices = dirichlet_partition(
        train_dataset, num_clients=args.num_clients, alpha=args.alpha, seed=args.seed
    )
    for client_id, indices in enumerate(client_indices):
        print(f"  client {client_id}: {len(indices)} samples")

    test_loader = DataLoader(test_dataset, batch_size=128)

    # ---- 2. Flower ClientApp: builds one client per virtual "supernode" --
    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        train_subset = make_client_subset(train_dataset, client_indices[partition_id])
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

        client = FedEntropyClient(
            train_loader=train_loader,
            num_classes=NUM_CLASSES,
            local_epochs=args.local_epochs,
            lr=args.lr,
        )
        return client.to_client()

    client_app = ClientApp(client_fn=client_fn)

    # ---- 3. Flower ServerApp: FedEntropy strategy + centralized eval -----
    def evaluate_fn(server_round, parameters, config):
        """
        Called by Flower after each round's aggregation, on the server,
        using the shared test set. Mirrors `test()` in the original
        codebase, which is run once per round on the aggregated model.
        """
        net = SimpleCNN(num_classes=NUM_CLASSES)
        set_weights(net, parameters)
        loss, accuracy = evaluate_model(net, test_loader)
        print(f"[Round {server_round}] global model -> loss={loss:.4f}, accuracy={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    def server_fn(context):
        strategy = FedEntropyStrategy(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=0.0,  # centralized evaluation (above) is used instead
            min_fit_clients=max(1, int(args.num_clients * args.fraction_fit)),
            min_available_clients=args.num_clients,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=lambda metrics: {},  # nothing to aggregate; silences a log warning
        )
        config = ServerConfig(num_rounds=args.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server_app = ServerApp(server_fn=server_fn)

    # ---- 4. Run the simulation --------------------------------------------
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )


if __name__ == "__main__":
    main()
