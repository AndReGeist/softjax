"""Plot MNIST sorting experiment results from CSV files.

Reads the curves and results CSVs produced by mnist.py and generates:
1. Training loss and validation accuracy curves (one line per experiment config)
2. A scatter plot of test_acc_em vs test_acc_ew across experiment configs
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def run_label(row):
    """Build a short label from experiment identity columns."""
    return f"{row['method']} s={row['softness']} n={row['num_compare']}"


def plot_curves(curves_df, ax_loss, ax_val):
    """Plot train loss and validation accuracy curves."""
    groups = curves_df.groupby(["method", "softness", "num_compare", "mode", "seed"])

    for key, group in groups:
        label = run_label(dict(zip(["method", "softness", "num_compare", "mode", "seed"], key)))
        ax_loss.semilogy(group["step"], group["train_loss"], label=label, alpha=0.8)

        val = group.dropna(subset=["val_acc_ew"])
        if not val.empty:
            ax_val.plot(val["step"], val["val_acc_ew"], linestyle="--", marker="o", markersize=3, label=label, alpha=0.8)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(fontsize="small")

    ax_val.set_xlabel("Step")
    ax_val.set_ylabel("Validation Element-wise match")
    ax_val.set_title("Validation Accuracy")
    ax_val.legend(fontsize="small")


def plot_scatter(results_df, ax):
    """Scatter plot of test_acc_em (x) vs test_acc_ew (y)."""
    results_df = results_df.dropna(subset=["test_acc_em", "test_acc_ew"])

    for _, row in results_df.iterrows():
        label = run_label(row)
        ax.scatter(row["test_acc_em"], row["test_acc_ew"], s=60, zorder=3)
        ax.annotate(label, (row["test_acc_em"], row["test_acc_ew"]),
                    textcoords="offset points", xytext=(6, 4), fontsize="small")

    ax.set_xlabel("Test Exact Match (acc_em)")
    ax.set_ylabel("Test Element-wise (acc_ew)")
    ax.set_title("Test Accuracy: Exact Match vs Element-wise")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y = x")
    ax.legend(fontsize="small")


def main():
    parser = argparse.ArgumentParser(description="Plot MNIST sorting experiment results")
    parser.add_argument("--results_csv", type=str, default="docs/examples/mnist/results.csv")
    parser.add_argument("--curves_csv", type=str, default="docs/examples/mnist/curves.csv")
    parser.add_argument("--out", type=str, default="docs/examples/mnist/mnist_plot.png", help="Save figure to file instead of showing")
    args = parser.parse_args()

    curves_df = pd.read_csv(args.curves_csv)
    results_df = pd.read_csv(args.results_csv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_curves(curves_df, axes[0], axes[1])
    plot_scatter(results_df, axes[2])
    fig.tight_layout()

    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
