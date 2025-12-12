"""
Q2 (Section 11.8): Handwritten Digit Recognition from Digital Images.
Implementation: single-hidden-layer MLP (784 -> 128 ReLU -> 10) using numpy.
Dataset: MNIST idx files (data/MNIST-dataset).
Outputs: models/q2_mlp.npz and metrics printed to stdout.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import numpy as np


@dataclasses.dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _load_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    images = data.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    return images


def _load_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels


def load_dataset(data_dir: Path, val_size: int = 5000) -> Dataset:
    train_images = _load_images(data_dir / "train-images-idx3-ubyte")
    train_labels = _load_labels(data_dir / "train-labels-idx1-ubyte")
    test_images = _load_images(data_dir / "t10k-images-idx3-ubyte")
    test_labels = _load_labels(data_dir / "t10k-labels-idx1-ubyte")

    x_val = train_images[-val_size:]
    y_val = train_labels[-val_size:]
    x_train = train_images[:-val_size]
    y_train = train_labels[:-val_size]

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=test_images,
        y_test=test_labels,
    )


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.size, num_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train_mlp(
    dataset: Dataset,
    epochs: int = 5,
    lr: float = 0.05,
    hidden: int = 128,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_features = dataset.x_train.shape
    num_classes = 10
    rng = np.random.default_rng(123)

    W1 = rng.normal(0, 0.05, size=(n_features, hidden)).astype(np.float32)
    b1 = np.zeros((hidden,), dtype=np.float32)
    W2 = rng.normal(0, 0.05, size=(hidden, num_classes)).astype(np.float32)
    b2 = np.zeros((num_classes,), dtype=np.float32)

    y_train_oh = one_hot(dataset.y_train, num_classes)

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n_samples)
        x_shuffled = dataset.x_train[idx]
        y_shuffled = y_train_oh[idx]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = x_shuffled[start:end]
            yb = y_shuffled[start:end]

            # Forward
            z1 = xb @ W1 + b1
            a1 = relu(z1)
            logits = a1 @ W2 + b2
            probs = softmax(logits)

            # Backward
            grad_logits = (probs - yb) / xb.shape[0]
            grad_W2 = a1.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)

            grad_a1 = grad_logits @ W2.T
            grad_z1 = grad_a1 * (z1 > 0)
            grad_W1 = xb.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1

        train_acc = accuracy(dataset.x_train, dataset.y_train, W1, b1, W2, b2)
        val_acc = accuracy(dataset.x_val, dataset.y_val, W1, b1, W2, b2)
        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    return W1, b1, W2, b2


def predict(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    a1 = relu(x @ W1 + b1)
    logits = a1 @ W2 + b2
    probs = softmax(logits)
    return probs.argmax(axis=1)


def accuracy(
    x: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> float:
    preds = predict(x, W1, b1, W2, b2)
    return (preds == y).mean()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "MNIST-dataset"
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    dataset = load_dataset(data_dir)
    W1, b1, W2, b2 = train_mlp(dataset)

    test_acc = accuracy(dataset.x_test, dataset.y_test, W1, b1, W2, b2)
    print(f"Test accuracy: {test_acc:.4f}")

    np.savez(models_dir / "q2_mlp.npz", W1=W1, b1=b1, W2=W2, b2=b2, test_acc=test_acc)
    print(f"Model saved to {models_dir / 'q2_mlp.npz'}")


if __name__ == "__main__":
    main()

