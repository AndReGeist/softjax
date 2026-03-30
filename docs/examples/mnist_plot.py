"""Plot MNIST sorting experiment results from CSV files.

Reads the curves and results CSVs produced by mnist.py and generates:
1. Training loss and validation accuracy curves (one line per experiment config)
2. A scatter plot of test metrics across experiment configs

Supports both task="sort" and task="quantile" metrics automatically.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def detect_task(df):
    """Detect task type from DataFrame columns."""
    if "task" in df.columns:
        tasks = df["task"].dropna().unique()
        if len(tasks) == 1:
            return tasks[0]
    if "val_acc_ew" in df.columns and df["val_acc_ew"].notna().any():
        return "sort"
    if "test_acc_ew" in df.columns and df["test_acc_ew"].notna().any():
        return "sort"
    return "quantile"


def run_label(row):
    """Build a short label from experiment identity columns."""
    return f"{row['method']} s={row['softness']} n={row['num_compare']}"


def plot_curves(curves_df, ax_loss, ax_val, task):
    """Plot train loss and validation accuracy curves."""
    groups = curves_df.groupby(["method", "softness", "num_compare", "mode", "seed"])

    if task == "sort":
        val_col, val_label = "val_acc_ew", "Validation Element-wise Accuracy"
    else:
        val_col, val_label = "val_spearman", "Validation Spearman Correlation"

    for key, group in groups:
        label = run_label(dict(zip(["method", "softness", "num_compare", "mode", "seed"], key)))
        ax_loss.semilogy(group["step"], group["train_loss"], label=label, alpha=0.8)

        val = group.dropna(subset=[val_col])
        if not val.empty:
            ax_val.plot(val["step"], val[val_col], linestyle="--", marker="o", markersize=3, label=label, alpha=0.8)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(fontsize="small")

    ax_val.set_xlabel("Step")
    ax_val.set_ylabel(val_label)
    ax_val.set_title(val_label)
    ax_val.legend(fontsize="small")


def plot_scatter(results_df, ax, task):
    """Scatter plot of test metrics."""
    if task == "sort":
        x_col, y_col = "test_acc_em", "test_acc_ew"
        x_label, y_label = "Test Exact Match (acc_em)", "Test Element-wise (acc_ew)"
        title = "Test Accuracy: Exact Match vs Element-wise"
    else:
        x_col, y_col = "test_mse", "test_spearman"
        x_label, y_label = r"Test MSE ($\times 10^{-4}$)", "Test Spearman Correlation"
        title = "Test: MSE vs Spearman"

    results_df = results_df.dropna(subset=[x_col, y_col])

    for _, row in results_df.iterrows():
        label = run_label(row)
        x_val = row[x_col] * 1e-4 if task == "quantile" else row[x_col]
        y_val = row[y_col]
        ax.scatter(x_val, y_val, s=60, zorder=3)
        ax.annotate(label, (x_val, y_val),
                    textcoords="offset points", xytext=(6, 4), fontsize="small")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if task == "sort":
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

    task = detect_task(curves_df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_curves(curves_df, axes[0], axes[1], task)
    plot_scatter(results_df, axes[2], task)
    fig.tight_layout()

    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
