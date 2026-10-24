"""Метрики качества вероятностных прогнозов исходов П1/X/П2."""
from __future__ import annotations

import numpy as np


def rps(probs: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score — стандарт для футбола (учитывает порядок П1 < X < П2). Меньше — лучше."""
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(probs, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, 1) / 2))


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - np.eye(3)[y]) ** 2, 1)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(probs.argmax(1) == y))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Ошибка калибровки: насколько «60%» действительно сбываются в 60% случаев."""
    onehot = np.eye(3)[y]
    p, o = probs.ravel(), onehot.ravel()
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(p[mask].mean() - o[mask].mean())
    return float(total / len(p))


def report(probs: np.ndarray, y: np.ndarray) -> dict:
    return {"RPS": rps(probs, y), "LogLoss": log_loss(probs, y), "Brier": brier(probs, y),
            "Accuracy": accuracy(probs, y), "ECE": ece(probs, y)}
