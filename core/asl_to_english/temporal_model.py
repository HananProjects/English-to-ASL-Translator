from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class LoadedTemporalModel:
    model_type: str
    labels: List[str]
    seq_len: int
    feature_dim: int
    mean: np.ndarray
    std: np.ndarray
    params: Dict[str, np.ndarray]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
    logits = np.clip(logits, -60.0, 60.0)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-9)


def normalize_input(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    flat = X.reshape(X.shape[0], -1).astype(np.float32)
    Xn = (flat - mean) / std
    Xn = np.clip(np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
    return Xn.reshape(X.shape[0], X.shape[1], X.shape[2]).astype(np.float32)


def temporal_conv_forward(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
):
    n, seq_len, feat_dim = X.shape
    hidden_dim, kernel_size, kernel_feat_dim = W1.shape
    if kernel_feat_dim != feat_dim:
        raise ValueError("Temporal conv input feature size mismatch")
    pad = kernel_size // 2
    Xpad = np.pad(X, ((0, 0), (pad, pad), (0, 0)), mode="constant")
    conv = np.zeros((n, seq_len, hidden_dim), dtype=np.float32)
    for k in range(kernel_size):
        conv += Xpad[:, k:k + seq_len, :] @ W1[:, k, :].T
    conv += b1.reshape(1, 1, hidden_dim)
    act = np.maximum(conv, 0.0)
    pooled = act.mean(axis=1)
    logits = pooled @ W2 + b2
    cache = {
        "X": X,
        "Xpad": Xpad,
        "conv": conv,
        "act": act,
        "pooled": pooled,
    }
    return logits, cache


def temporal_conv_backward(
    grad_logits: np.ndarray,
    cache: Dict[str, np.ndarray],
    W1: np.ndarray,
    W2: np.ndarray,
):
    X = cache["X"]
    Xpad = cache["Xpad"]
    conv = cache["conv"]
    act = cache["act"]

    n, seq_len, feat_dim = X.shape
    hidden_dim, kernel_size, _ = W1.shape

    dW2 = cache["pooled"].T @ grad_logits
    db2 = grad_logits.sum(axis=0)

    dpooled = grad_logits @ W2.T
    dact = np.repeat((dpooled / seq_len)[:, None, :], seq_len, axis=1)
    dconv = dact * (conv > 0.0).astype(np.float32)

    dW1 = np.zeros_like(W1)
    db1 = dconv.sum(axis=(0, 1))
    for k in range(kernel_size):
        x_slice = Xpad[:, k:k + seq_len, :]
        dW1[:, k, :] = np.einsum("nth,ntf->hf", dconv, x_slice)

    return dW1, db1, dW2, db2


def save_linear_model(
    path,
    W: np.ndarray,
    b: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    labels: List[str],
    seq_len: int,
    feature_dim: int,
) -> None:
    np.savez_compressed(
        path,
        model_type=np.asarray("linear"),
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        labels=np.asarray(labels),
        seq_len=np.asarray(seq_len, dtype=np.int32),
        feature_dim=np.asarray(feature_dim, dtype=np.int32),
    )


def save_temporal_conv_model(
    path,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    labels: List[str],
    seq_len: int,
    feature_dim: int,
) -> None:
    np.savez_compressed(
        path,
        model_type=np.asarray("temporal_cnn"),
        conv_W=W1.astype(np.float32),
        conv_b=b1.astype(np.float32),
        fc_W=W2.astype(np.float32),
        fc_b=b2.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        labels=np.asarray(labels),
        seq_len=np.asarray(seq_len, dtype=np.int32),
        feature_dim=np.asarray(feature_dim, dtype=np.int32),
    )


def load_sequence_model(path) -> LoadedTemporalModel:
    model = np.load(path, allow_pickle=True)
    model_type = str(model["model_type"]) if "model_type" in model else "linear"
    labels = [str(x).upper() for x in model["labels"].tolist()]
    seq_len = int(model["seq_len"])
    feature_dim = int(model["feature_dim"])
    mean = np.nan_to_num(model["mean"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    std = model["std"].astype(np.float32)
    std[std < 1e-6] = 1.0
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)

    params: Dict[str, np.ndarray]
    if model_type == "temporal_cnn":
        params = {
            "conv_W": np.clip(
                np.nan_to_num(model["conv_W"].astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            ),
            "conv_b": np.clip(
                np.nan_to_num(model["conv_b"].astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            ),
            "fc_W": np.clip(
                np.nan_to_num(model["fc_W"].astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            ),
            "fc_b": np.clip(
                np.nan_to_num(model["fc_b"].astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            ),
        }
    else:
        params = {
            "W": np.clip(
                np.nan_to_num(model["W"].astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            ),
            "b": np.clip(
                np.nan_to_num(model["b"].astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            ),
        }

    return LoadedTemporalModel(
        model_type=model_type,
        labels=labels,
        seq_len=seq_len,
        feature_dim=feature_dim,
        mean=mean,
        std=std,
        params=params,
    )


def predict_logits(model: LoadedTemporalModel, X: np.ndarray) -> np.ndarray:
    Xn = normalize_input(X, model.mean, model.std)
    if model.model_type == "temporal_cnn":
        logits, _ = temporal_conv_forward(
            Xn,
            model.params["conv_W"],
            model.params["conv_b"],
            model.params["fc_W"],
            model.params["fc_b"],
        )
        return logits

    flat = Xn.reshape(Xn.shape[0], -1)
    logits = flat @ model.params["W"] + model.params["b"]
    logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
    return np.clip(logits, -60.0, 60.0)
