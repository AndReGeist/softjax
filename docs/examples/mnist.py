"""Sort numbers consisting of MNIST digits.

Reproduces the multi-digit MNIST sorting experiment from the DiffSort paper
(Petersen et al., ICLR 2022) using JAX, Equinox, and SoftJAX.

A CNN learns to map concatenated multi-digit MNIST images to scalar scores.
A soft argsort produces a differentiable permutation matrix, trained via BCE
loss against the ground-truth ranking.
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MNIST sorting benchmark (SoftJAX)")
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("-n", "--num_compare", type=int, default=5)
    parser.add_argument("-i", "--num_steps", type=int, default=200_000)
    parser.add_argument("-e", "--eval_freq", type=int, default=1_000)
    parser.add_argument(
        "--softness", type=float, default=0.1, help="Softness (inverse of diffsort steepness)"
    )
    parser.add_argument("--mode", type=str, default="smooth", choices=["smooth", "c0", "c1", "c2"])
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
    softness = args.softness
    mode = args.mode
    method = args.method
    standardize = args.standardize

    @eqx.filter_jit
    def make_step(model, opt_state, data, targets):
        def loss_fn(model):
            scores = jax.vmap(jax.vmap(model))(data)  # (batch, num_compare)
            perm_pred = sj.argsort(
                scores,
                axis=-1,
                softness=softness,
                mode=mode,
                method=method,
                standardize=standardize,
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
        scores = jax.vmap(jax.vmap(model))(data)  # (batch, num_compare)

        pred_order = jnp.argsort(scores, axis=-1)
        true_order = jnp.argsort(targets, axis=-1)
        acc = pred_order == true_order

        acc_em = jnp.all(acc, axis=-1).astype(jnp.float32).mean()
        acc_ew = acc.astype(jnp.float32).mean()

        # EM5: exact match restricted to first 5 elements.
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

    # --- Training loop ---
    best_valid_acc = 0.0
    test_acc = None
    curve_records = []

    for iter_idx, (data, targets) in tqdm(
        enumerate(load_n(train_loader, args.num_steps)),
        desc="Training",
        total=args.num_steps,
    ):
        data, targets = data.numpy(), targets.numpy()
        model, opt_state, loss = make_step(model, opt_state, data, targets)

        record = {"step": iter_idx, "train_loss": loss.item(), "val_acc_em": None, "val_acc_ew": None, "val_acc_em5": None}

        if (iter_idx + 1) % args.eval_freq == 0:
            valid_acc = evaluate_loader(model, valid_loader)
            print(f"{iter_idx} valid {valid_acc}")
            record["val_acc_em"] = valid_acc["acc_em"]
            record["val_acc_ew"] = valid_acc["acc_ew"]
            record["val_acc_em5"] = valid_acc["acc_em5"]

            if valid_acc["acc_em5"] > best_valid_acc:
                best_valid_acc = valid_acc["acc_em5"]
                test_acc = evaluate_loader(model, test_loader)
                print(f"{iter_idx} test  {test_acc}")

        curve_records.append(record)

    print(f"final test {test_acc}")

    # --- Ensure output directory exists ---
    os.makedirs(os.path.dirname(args.curves_csv), exist_ok=True)

    # --- Save curves CSV ---
    curves_df = pd.DataFrame(curve_records)
    curves_df["method"] = method
    curves_df["mode"] = mode
    curves_df["softness"] = softness
    curves_df["num_compare"] = args.num_compare
    curves_df["seed"] = args.seed
    header = not os.path.exists(args.curves_csv)
    curves_df.to_csv(args.curves_csv, mode="a", header=header, index=False)

    # --- Save results CSV ---
    result = {
        "method": method,
        "mode": mode,
        "softness": softness,
        "standardize": standardize,
        "num_compare": args.num_compare,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "test_acc_em": test_acc["acc_em"] if test_acc else None,
        "test_acc_ew": test_acc["acc_ew"] if test_acc else None,
        "test_acc_em5": test_acc["acc_em5"] if test_acc else None,
        "best_valid_acc_em5": best_valid_acc,
    }
    results_df = pd.DataFrame([result])
    header = not os.path.exists(args.results_csv)
    results_df.to_csv(args.results_csv, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()
