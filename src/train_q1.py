"""
Q1 (Section 10.9): Handwritten Digit Recognition from Digital Images.
Implementation: softmax regression (784 -> 10) using numpy.
Dataset: MNIST idx files (data/MNIST-dataset).
Outputs: models/q1_softmax.npz and metrics printed to stdout.
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

    # Ayırma: son val_size doğrulama, geri kalanı eğitim
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


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax(
    dataset: Dataset,
    epochs: int = 5,
    lr: float = 0.1,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = dataset.x_train.shape
    num_classes = 10
    rng = np.random.default_rng(42)

    W = rng.normal(0, 0.01, size=(n_features, num_classes)).astype(np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)

    y_train_oh = one_hot(dataset.y_train, num_classes)

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n_samples)
        x_shuffled = dataset.x_train[idx]
        y_shuffled = y_train_oh[idx]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = x_shuffled[start:end]
            yb = y_shuffled[start:end]

            logits = xb @ W + b
            probs = softmax(logits)
            grad_logits = (probs - yb) / xb.shape[0]
            grad_W = xb.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)

            W -= lr * grad_W
            b -= lr * grad_b

        train_acc = accuracy(dataset.x_train, dataset.y_train, W, b)
        val_acc = accuracy(dataset.x_val, dataset.y_val, W, b)
        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    return W, b


def accuracy(x: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    preds = predict(x, W, b)
    return (preds == y).mean()


def predict(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits = x @ W + b
    probs = softmax(logits)
    return probs.argmax(axis=1)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "MNIST-dataset"
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    dataset = load_dataset(data_dir)
    W, b = train_softmax(dataset)

    test_acc = accuracy(dataset.x_test, dataset.y_test, W, b)
    print(f"Test accuracy: {test_acc:.4f}")

    np.savez(models_dir / "q1_softmax.npz", W=W, b=b, test_acc=test_acc)
    print(f"Model saved to {models_dir / 'q1_softmax.npz'}")


if __name__ == "__main__":
    main()

