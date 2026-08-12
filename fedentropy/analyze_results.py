"""
analyze_results.py
-------------------
Reads results/fedavg.csv and results/fedentopt.csv (produced by
run_experiment.py) and:
  1. Prints a summary table (final accuracy, mean of last-10-round accuracy,
     absolute improvement of FedEntOpt over FedAvg -- matching how the paper
     reports numbers).
  2. Saves a comparison plot (accuracy vs. round, loss vs. round) to
     results/comparison.png.

Usage:
    python analyze_results.py --results-dir ./results
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="./results")
    p.add_argument("--last-n", type=int, default=10, help="Rounds averaged for final accuracy")
    return p.parse_args()


def load(results_dir, name):
    path = os.path.join(results_dir, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Round 0 is the initial (random-init) evaluation; drop it for training curves.
    return df[df["round"] > 0].reset_index(drop=True)


def summarize(df, name, last_n):
    tail = df.tail(last_n)
    print(f"\n{name}")
    print(f"  final round accuracy      : {df['accuracy'].iloc[-1]*100:.2f}%")
    print(f"  mean acc (last {last_n:>2d} rounds) : {tail['accuracy'].mean()*100:.2f}% "
          f"(+/- {tail['accuracy'].std()*100:.2f})")
    print(f"  mean loss (last {last_n:>2d} rounds): {tail['loss'].mean():.4f}")
    return tail["accuracy"].mean()


def main():
    args = parse_args()
    fedavg = load(args.results_dir, "fedavg")
    fedentopt = load(args.results_dir, "fedentopt")

    if fedavg is None and fedentopt is None:
        raise SystemExit(f"No result CSVs found in {args.results_dir}. Run run_experiment.py first.")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    acc_avg = summarize(fedavg, "FedAvg", args.last_n) if fedavg is not None else None
    acc_ent = summarize(fedentopt, "FedEntOpt", args.last_n) if fedentopt is not None else None
    if acc_avg is not None and acc_ent is not None:
        print(f"\nFedEntOpt improvement over FedAvg: {(acc_ent - acc_avg) * 100:+.2f} percentage points")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for df, label, color in [(fedavg, "FedAvg", "tab:blue"), (fedentopt, "FedEntOpt", "tab:orange")]:
        if df is None:
            continue
        axes[0].plot(df["round"], df["accuracy"] * 100, label=label, color=color)
        axes[1].plot(df["round"], df["loss"], label=label, color=color)

    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Centralized test accuracy (%)")
    axes[0].set_title("Accuracy vs. round")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Centralized test loss")
    axes[1].set_title("Loss vs. round")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("FedAvg vs. FedEntOpt on CIFAR-10 (ResNet-18, label skew)")
    fig.tight_layout()

    out_path = os.path.join(args.results_dir, "comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
