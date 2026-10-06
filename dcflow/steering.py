"""Structural crowding selection at the sequence activation boundary."""

import numpy as np
from scipy.special import logsumexp


def rmsd(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b) ** 2, axis=-1))))


def select(backbones, count, seed, sigma=1.75, temperature=2.5, epsilon=1e-8):
    xyz = np.asarray(backbones, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected finite [K,L,3] C-alpha coordinates")
    if not 1 <= count <= len(xyz) or min(sigma, temperature, epsilon) <= 0:
        raise ValueError("Invalid structural selection parameters")
    distances = np.zeros((len(xyz), len(xyz)))
    for i in range(len(xyz)):
        for j in range(i):
            distances[i, j] = distances[j, i] = rmsd(xyz[i], xyz[j])
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(xyz)))]
    log_kernel = -(distances**2) / (2 * sigma**2)
    while len(chosen) < count:
        crowd = logsumexp(log_kernel[:, chosen], axis=1) - np.log(len(chosen))
        score = -np.logaddexp(np.log(epsilon), crowd) / temperature
        score[chosen] = -np.inf
        chosen.append(int(rng.choice(len(xyz), p=np.exp(score - logsumexp(score)))))
    return chosen


def quality_select(scores, count, seed, temperature=0.1):
    scores = np.asarray(scores, dtype=float)
    if (
        not np.isfinite(scores).all()
        or not 1 <= count <= len(scores)
        or temperature <= 0
    ):
        raise ValueError("Invalid quality-selection input")
    logits = (scores - scores.max()) / temperature
    # Sequential draws avoid numerical loss of positive support at small temperatures.
    rng, selected = np.random.default_rng(seed), []
    for _ in range(count):
        p = np.exp(logits - logsumexp(logits))
        index = int(rng.choice(len(scores), p=p))
        selected.append(index)
        logits[index] = -np.inf
    return selected
