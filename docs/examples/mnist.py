"""Sort or quantile-regress numbers consisting of MNIST digits.

Reproduces the multi-digit MNIST sorting and quantile regression experiments
from the DiffSort / SoftSort papers using JAX, Equinox, and SoftJAX.

A CNN learns to map concatenated multi-digit MNIST images to scalar scores.
- **sort** task: A soft argsort produces a differentiable permutation matrix,
  trained via BCE loss against the ground-truth ranking.
- **quantile** task: A soft quantile extracts a differentiable quantile value
  (e.g. median) from the scores, trained via MSE against the true quantile.
"""

import argparse
import os
import random

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

import softjax as sj


# ---------------------------------------------------------------------------
# Data loading (PyTorch) -- adapted from diffsort/experiments/datasets/dataset.py
# ---------------------------------------------------------------------------


class MultiDigitDataset(Dataset):
    """Generates multi-digit MNIST samples.

    For each sample, ``num_compare`` composite images are created by
    concatenating ``num_digits`` randomly chosen MNIST digit images along the
    width axis.  The label for each composite image is a multi-digit number
    (e.g. digits 3, 7, 4, 2 -> label 3742).
    
    Note that the implementation of this data loader is quite lazy. We pick random 
    sequences with REPLACEMENT, so the same sequence may appear multiple times in an epoch, 
    while some sequences may never appear.  This is done to avoid the combinatorial explosion of possible sequences, 
    which would make it infeasible to pre-generate a fixed dataset of all possible sequences.
    """

    def __init__(self, images, labels, num_digits, num_compare, seed=0, determinism=True):
        super().__init__()
        self.images = images
        self.labels = labels
        self.num_digits = num_digits
        self.num_compare = num_compare
        self.seed = seed
        self.rand_state = None
        self.determinism = determinism
        if determinism:
            self.reset_rand_state()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if self.determinism:
            prev_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self.rand_state)

        images = []
        labels_ = torch.zeros(self.num_compare, dtype=torch.float32)
        for digit_idx in range(self.num_digits):
            ids = torch.randint(len(self), (self.num_compare,))
            images.append(self.images[ids].type(torch.float32) / 255.0)
            labels_ = labels_ + 10.0 ** (self.num_digits - 1 - digit_idx) * self.labels[ids]

        images = torch.cat(images, dim=-1)  # (num_compare, 1, 28, num_digits*28)

        if self.determinism:
            self.rand_state = torch.random.get_rng_state()
            torch.random.set_rng_state(prev_state)

        return images, labels_

    def reset_rand_state(self):
        prev_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        self.rand_state = torch.random.get_rng_state()
        torch.random.set_rng_state(prev_state)


class MultiDigitSplits:
    """Train / validation / test splits for MNIST multi-digit sorting."""

    def __init__(self, num_digits=4, num_compare=5, seed=0, deterministic_data_loader=True):
        self.deterministic_data_loader = deterministic_data_loader

        trva = datasets.MNIST(root="./data-mnist", download=True)
        xtr = trva.data[:55000].view(-1, 1, 28, 28)
        ytr = trva.targets[:55000]
        xva = trva.data[55000:].view(-1, 1, 28, 28)
        yva = trva.targets[55000:]

        te = datasets.MNIST(root="./data-mnist", train=False, download=True)
        xte = te.data.view(-1, 1, 28, 28)
        yte = te.targets

        kw = dict(num_digits=num_digits, num_compare=num_compare, seed=seed)
        self.train_dataset = MultiDigitDataset(xtr, ytr, determinism=deterministic_data_loader, **kw)
        self.valid_dataset = MultiDigitDataset(xva, yva, **kw)
        self.test_dataset = MultiDigitDataset(xte, yte, **kw)

    def get_train_loader(self, batch_size, **kwargs):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=4 if not self.deterministic_data_loader else 0,
            shuffle=True,
            **kwargs,
        )

    def get_valid_loader(self, batch_size, **kwargs):
        return DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


# ---------------------------------------------------------------------------
# Model (Equinox)
# ---------------------------------------------------------------------------


class MultiDigitMNISTNet(eqx.Module):
    """CNN that maps a concatenated multi-digit image to a scalar score.

    Architecture (for default ``n_digits=4``):
        (1, 28, 112) -> Conv(1->32, 5x5, pad=2) -> ReLU -> MaxPool(2)
        -> Conv(32->64, 5x5, pad=2) -> ReLU -> MaxPool(2)
        -> Flatten(12544) -> Linear(64) -> ReLU -> Linear(1)

    Operates on a **single** image; use ``jax.vmap`` for batches.
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, n_digits: int = 4, *, key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, key=k1)
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, key=k2)
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = eqx.nn.Linear(n_digits * 7 * 7 * 64, 64, key=k3)
        self.fc2 = eqx.nn.Linear(64, 1, key=k4)

    def __call__(self, x):
        x = self.pool(jax.nn.relu(self.conv1(x)))
        x = self.pool(jax.nn.relu(self.conv2(x)))
        x = jnp.ravel(x)
        x = jax.nn.relu(self.fc1(x))
        return self.fc2(x).squeeze()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_n(loader, n):
    """Cycle through *loader* yielding exactly *n* batches."""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def bce_loss(pred, target):
    """Binary cross-entropy for probability inputs (not logits)."""
    eps = 1e-7
    pred = jnp.clip(pred, eps, 1.0 - eps)
    return -jnp.mean(target * jnp.log(pred) + (1.0 - target) * jnp.log(1.0 - pred))


def spearman_correlation(pred, true):
    """Spearman rank correlation between two 1-D numpy arrays."""
    def _rank(x):
        return np.argsort(np.argsort(x)).astype(np.float64)
    n = len(pred)
    if n < 2:
        return float("nan")
    d = _rank(pred) - _rank(true)
    return 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MNIST sorting / quantile benchmark (SoftJAX)")
    parser.add_argument("--task", type=str, default="sort", choices=["sort", "quantile"])
    parser.add_argument("--quantile", type=float, default=0.5, help="Quantile q in [0,1]; 0.5 = median")
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("-n", "--num_compare", type=int, default=5)
    parser.add_argument("-i", "--num_steps", type=int, default=200_000)
    parser.add_argument("-e", "--eval_freq", type=int, default=1_000)
    parser.add_argument(
        "--softness", type=float, default=0.1, help="Softness (inverse of diffsort steepness)"
    )
    parser.add_argument("--mode", type=str, default="smooth", choices=["hard", "smooth", "c0", "c1", "c2"])
    parser.add_argument(
        "--method",
        type=str,
        default="sorting_network",
        choices=["sorting_network", "neuralsort", "softsort", "ot"],
    )
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-l", "--lr", type=float, default=3.0e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_csv", type=str, default="docs/examples/mnist/results.csv")
    parser.add_argument("--curves_csv", type=str, default="docs/examples/mnist/curves.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    key = jrandom.PRNGKey(args.seed)

    # --- Data ---
    splits = MultiDigitSplits(num_compare=args.num_compare, seed=args.seed)
    loader_kw = dict(batch_size=args.batch_size, drop_last=True)
    train_loader = splits.get_train_loader(**loader_kw)
    valid_loader = splits.get_valid_loader(**loader_kw)
    test_loader = splits.get_test_loader(**loader_kw)

    # --- Model and optimizer ---
    key, model_key = jrandom.split(key)
    model = MultiDigitMNISTNet(key=model_key)
    optim = optax.adam(learning_rate=args.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Capture config in closures for the JIT-compiled functions.
    task = args.task
    quantile = args.quantile
    softness = args.softness
    mode = args.mode
    method = args.method
    standardize = args.standardize

    if task == "sort":
        @eqx.filter_jit
        def make_step(model, opt_state, data, targets):
            def loss_fn(model):
                """BCE loss between predicted and true permutation matrices."""
                scores = jax.vmap(jax.vmap(model))(data)  # (batch, num_compare)
                perm_pred = sj.argsort(
                    scores, axis=-1, softness=softness, mode=mode,
                    method=method, standardize=standardize,
                )  # (batch, num_compare, num_compare)
                num_compare = targets.shape[-1]
                perm_gt = jax.nn.one_hot(jnp.argsort(targets, axis=-1), num_compare)
                return bce_loss(perm_pred, perm_gt)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, new_opt_state, loss

        @eqx.filter_jit
        def evaluate_batch(model, data, targets):
            scores = jax.vmap(jax.vmap(model))(data)
            pred_order = jnp.argsort(scores, axis=-1)
            true_order = jnp.argsort(targets, axis=-1)
            acc = pred_order == true_order
            acc_em = jnp.all(acc, axis=-1).astype(jnp.float32).mean()
            acc_ew = acc.astype(jnp.float32).mean()
            scores5 = scores[:, :5]
            targets5 = targets[:, :5]
            acc5 = jnp.argsort(scores5, axis=-1) == jnp.argsort(targets5, axis=-1)
            acc_em5 = jnp.all(acc5, axis=-1).astype(jnp.float32).mean()
            return acc_em, acc_ew, acc_em5

        def evaluate_loader(model, loader):
            results = []
            for data, targets in loader:
                data, targets = data.numpy(), targets.numpy()
                em, ew, em5 = evaluate_batch(model, data, targets)
                results.append(dict(acc_em=em.item(), acc_ew=ew.item(), acc_em5=em5.item()))
            return {k: np.mean([d[k] for d in results]) for k in results[0]}

        best_val_score = 0.0
        val_metric_key = "acc_em5"
        higher_is_better = True
    elif task == "quantile":
        @eqx.filter_jit
        def make_step(model, opt_state, data, targets):
            def loss_fn(model):
                """MSE between predicted and true quantiles."""
                scores = jax.vmap(jax.vmap(model))(data)
                pred_q = sj.quantile(
                    scores, q=quantile, axis=-1, softness=softness,
                    mode=mode, method=method, standardize=standardize,
                )
                true_q = jnp.quantile(targets, q=quantile, axis=-1)
                return jnp.mean((pred_q - true_q) ** 2)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, new_opt_state, loss

        @eqx.filter_jit
        def evaluate_batch(model, data, targets):
            scores = jax.vmap(jax.vmap(model))(data)
            pred_q = sj.quantile(scores, q=quantile, axis=-1, mode="hard")
            true_q = jnp.quantile(targets, q=quantile, axis=-1)
            mse = jnp.mean((pred_q - true_q) ** 2)
            return mse, pred_q, true_q

        def evaluate_loader(model, loader):
            all_pred, all_true, mses = [], [], []
            for data, targets in loader:
                data, targets = data.numpy(), targets.numpy()
                mse, pred_q, true_q = evaluate_batch(model, data, targets)
                mses.append(mse.item())
                all_pred.append(np.asarray(pred_q))
                all_true.append(np.asarray(true_q))
            return {
                "mse": np.mean(mses),
                "spearman": spearman_correlation(np.concatenate(all_pred), np.concatenate(all_true)),
            }

        best_val_score = float("inf")
        val_metric_key = "mse"
        higher_is_better = False
    # --- Training loop ---
    test_metrics = None
    curve_records = []

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_steps)),
        desc="Training",
        total=args.num_steps,
    )
    for iter_idx, (data, targets) in pbar:
        data, targets = data.numpy(), targets.numpy()
        model, opt_state, loss = make_step(model, opt_state, data, targets)

        record = {"step": iter_idx, "train_loss": loss.item()}

        if (iter_idx + 1) % args.eval_freq == 0:
            valid_metrics = evaluate_loader(model, valid_loader)
            tqdm.write(f"{iter_idx} valid {valid_metrics}")
            if task == "sort":
                pbar.set_postfix(loss=f"{loss.item():.4f}", em=f"{valid_metrics['acc_em']:.3f}", ew=f"{valid_metrics['acc_ew']:.3f}")
                record.update(val_acc_em=valid_metrics["acc_em"], val_acc_ew=valid_metrics["acc_ew"], val_acc_em5=valid_metrics["acc_em5"])
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{valid_metrics['mse']:.4f}", spear=f"{valid_metrics['spearman']:.3f}")
                record.update(val_mse=valid_metrics["mse"], val_spearman=valid_metrics["spearman"])

            val_score = valid_metrics[val_metric_key]
            improved = val_score > best_val_score if higher_is_better else val_score < best_val_score
            if improved:
                best_val_score = val_score
                test_metrics = evaluate_loader(model, test_loader)
                tqdm.write(f"{iter_idx} test  {test_metrics}")

        curve_records.append(record)

    print(f"final test {test_metrics}")

    for path in (args.curves_csv, args.results_csv):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # --- Save curves CSV ---
    curves_df = pd.DataFrame(curve_records)
    curves_df["task"] = task
    curves_df["quantile"] = quantile if task == "quantile" else None
    curves_df["method"] = method
    curves_df["mode"] = mode
    curves_df["softness"] = softness
    curves_df["num_compare"] = args.num_compare
    curves_df["seed"] = args.seed
    header = not os.path.exists(args.curves_csv)
    curves_df.to_csv(args.curves_csv, mode="a", header=header, index=False)

    # --- Save results CSV ---
    result = {
        "task": task,
        "quantile": quantile if task == "quantile" else None,
        "method": method,
        "mode": mode,
        "softness": softness,
        "standardize": standardize,
        "num_compare": args.num_compare,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        # Sort metrics
        "test_acc_em": test_metrics.get("acc_em") if test_metrics else None,
        "test_acc_ew": test_metrics.get("acc_ew") if test_metrics else None,
        "test_acc_em5": test_metrics.get("acc_em5") if test_metrics else None,
        "best_valid_acc_em5": best_val_score if task == "sort" else None,
        # Quantile metrics
        "test_mse": test_metrics.get("mse") if test_metrics else None,
        "test_spearman": test_metrics.get("spearman") if test_metrics else None,
        "best_valid_mse": best_val_score if task == "quantile" else None,
    }
    results_df = pd.DataFrame([result])
    header = not os.path.exists(args.results_csv)
    results_df.to_csv(args.results_csv, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()
