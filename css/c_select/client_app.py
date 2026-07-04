"""c_select: A Flower / PyTorch app."""

import random

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from c_select.pqc import simulate_client_pqc_overhead
from c_select.task import Net, load_data
from c_select.task import test as test_fn
from c_select.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def _model_bytes(model: torch.nn.Module) -> int:
    """Approximate serialized size of the model update, in bytes."""
    return sum(p.numel() * p.element_size() for p in model.state_dict().values())


def _simulated_device_profile(partition_id: int) -> dict:
    """Stand-in for real device telemetry: deterministic per-client heterogeneity.

    In a real deployment these would come from the OS / battery / NIC; here we
    seed on partition_id so each simulated client has a stable "device" across
    rounds, which is what makes selection scores meaningful to compare.
    """
    rng = random.Random(partition_id)
    return {
        "cpu_score": rng.uniform(0.2, 1.0),
        "bandwidth_mbps": rng.uniform(2.0, 50.0),
        "dropout_probability": rng.uniform(0.0, 0.3),
        "trust": rng.uniform(0.7, 1.0),
    }


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data and report PQC + device metrics."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    model_bytes = _model_bytes(model)
    pqc_metrics = simulate_client_pqc_overhead(model_bytes)
    device_profile = _simulated_device_profile(partition_id)

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "data_size": len(trainloader.dataset),
        "model_bytes": model_bytes,
        **device_profile,
        **pqc_metrics,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)