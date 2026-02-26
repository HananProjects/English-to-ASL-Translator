import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight ASL landmark classifier from NPZ splits."
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/datasets/asl_landmarks_v1",
        help="Directory containing train.npz/val.npz/test.npz and metadata.json",
    )
    parser.add_argument(
        "--output-model",
        default="models/asl_landmark_classifier_v1.npz",
        help="Where to save the trained model artifact.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    return X, y


def flatten_features(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def sanitize_features(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -20.0, 20.0)


def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (X - mean) / std
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Z, -8.0, 8.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    picked = probs[np.arange(len(y)), y]
    return float(-np.log(picked + eps).mean())


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    return float((pred == y).mean())


def one_hot(y: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((len(y), classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def evaluate(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray):
    if len(X) == 0:
        return {"loss": 0.0, "acc": 0.0}
    logits = X @ W + b
    probs = softmax(logits)
    return {
        "loss": cross_entropy(probs, y),
        "acc": accuracy(logits, y),
    }


def main():
    args = parse_args()
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    rng = np.random.default_rng(args.seed)

    dataset_dir = Path(args.dataset_dir)
    train_X, train_y = load_split(dataset_dir / "train.npz")
    val_X, val_y = load_split(dataset_dir / "val.npz")
    test_X, test_y = load_split(dataset_dir / "test.npz")

    with open(dataset_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    labels = metadata["labels"]
    num_classes = len(labels)

    train_X = flatten_features(train_X)
    val_X = flatten_features(val_X)
    test_X = flatten_features(test_X)
    train_X = sanitize_features(train_X)
    val_X = sanitize_features(val_X)
    test_X = sanitize_features(test_X)

    mean, std = standardize_fit(train_X)
    train_X = standardize_apply(train_X, mean, std)
    val_X = standardize_apply(val_X, mean, std) if len(val_X) else val_X
    test_X = standardize_apply(test_X, mean, std) if len(test_X) else test_X

    in_dim = train_X.shape[1]
    W = (rng.normal(0.0, 0.01, size=(in_dim, num_classes))).astype(np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)

    best = {
        "val_acc": -1.0,
        "W": W.copy(),
        "b": b.copy(),
    }

    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(len(train_X))
        X_shuf = train_X[indices]
        y_shuf = train_y[indices]

        for start in range(0, len(X_shuf), args.batch_size):
            end = start + args.batch_size
            xb = X_shuf[start:end]
            yb = y_shuf[start:end]
            if len(xb) == 0:
                continue

            logits = xb @ W + b
            probs = softmax(logits)
            target = one_hot(yb, num_classes)

            grad_logits = (probs - target) / len(xb)
            grad_W = xb.T @ grad_logits + args.weight_decay * W
            grad_b = grad_logits.sum(axis=0)
            grad_W = np.clip(grad_W, -1.0, 1.0)
            grad_b = np.clip(grad_b, -1.0, 1.0)
            grad_W = np.nan_to_num(grad_W, nan=0.0, posinf=0.0, neginf=0.0)
            grad_b = np.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0)

            W -= args.lr * grad_W
            b -= args.lr * grad_b
            W = np.clip(W, -5.0, 5.0)
            b = np.clip(b, -5.0, 5.0)

        train_metrics = evaluate(train_X, train_y, W, b)
        val_metrics = evaluate(val_X, val_y, W, b)
        if val_metrics["acc"] >= best["val_acc"]:
            best["val_acc"] = val_metrics["acc"]
            best["W"] = W.copy()
            best["b"] = b.copy()

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.3f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['acc']:.3f}"
            )

    final_W = best["W"]
    final_b = best["b"]
    test_metrics = evaluate(test_X, test_y, final_W, final_b)
    print(
        f"[test] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.3f}"
    )

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_model,
        W=final_W.astype(np.float32),
        b=final_b.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        labels=np.asarray(labels, dtype=object),
        seq_len=np.int64(metadata["seq_len"]),
        feature_dim=np.int64(metadata["feature_dim"]),
    )
    print(f"[saved] {output_model}")


if __name__ == "__main__":
    main()
