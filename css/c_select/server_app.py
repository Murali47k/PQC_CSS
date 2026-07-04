"""c_select: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from c_select.strategy import ScoredSelectionStrategy
from c_select.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config.get("fraction-train", 0.5)
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    scorer_name: str = context.run_config.get("selection-strategy", "pqc_aware")

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # PQC-aware (or baseline) client-selection strategy.
    # Swap `selection-strategy` in pyproject.toml's [tool.flwr.app.config] to
    # benchmark: "pqc_aware" | "random" | "data_size" | "loss_based" |
    # "resource_aware" | "trust_based".
    strategy = ScoredSelectionStrategy(
        scorer_name=scorer_name,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})