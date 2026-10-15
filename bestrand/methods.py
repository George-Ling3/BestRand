"""NumPy implementations of BestRand and uniform random sampling."""

import numpy as np


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rng(seed, stream):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return np.random.default_rng(np.random.SeedSequence([int(seed), *stream.encode("utf-8")]))


def _budget(k, n):
    k = _positive_int(k, "k")
    if k > n:
        raise ValueError(f"k must not exceed the number of items ({n})")
    return k


def select(probe_scores, k, seed=0, *, pool_multiplier=4, candidate_count=128):
    """Return k original column indices from finite [probe, item] scores.

    Only scores in a uniformly sampled pool determine the selected subset.
    Candidate order and independent RNG streams are fixed for reproducibility.
    Exact distance ties use the lexicographically smallest sorted index tuple.
    The returned columns keep candidate order; pass scores in that same order
    to estimate(). Continuous scores are used directly, without thresholding.
    """
    scores = np.asarray(probe_scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("probe_scores must be a non-empty finite [probe, item] matrix")
    k = _budget(k, scores.shape[1])
    multiplier = _positive_int(pool_multiplier, "pool_multiplier")
    count = _positive_int(candidate_count, "candidate_count")
    pool_size = min(multiplier * k, scores.shape[1])
    pool = _rng(seed, "pool").choice(scores.shape[1], pool_size, replace=False)
    target = scores[:, pool].mean(axis=1)
    rng = _rng(seed, "candidate_sampling")
    count = 1 if pool_size == k else count
    candidates = np.stack([rng.choice(pool_size, k, replace=False) for _ in range(count)])
    indices = pool[candidates]
    means = scores[:, indices].mean(axis=2).T
    delta = means - target[None, :]
    distances = np.square(delta).sum(axis=1)
    tied = np.flatnonzero(distances == distances.min())
    winner = min((int(i) for i in tied), key=lambda i: tuple(sorted(int(x) for x in indices[i])))
    return indices[winner].copy()


def random_select(n_items, k, seed=0):
    """Uniformly sample k distinct item indices, returned in sorted order."""
    n_items = _positive_int(n_items, "n_items")
    k = _budget(k, n_items)
    return np.sort(_rng(seed, "item_sampling").choice(n_items, k, replace=False))


def estimate(selected_scores):
    """Estimate full scores from a finite [target model, selected item] matrix."""
    values = np.asarray(selected_scores, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("selected_scores must be a non-empty finite [model, selected item] matrix")
    return values @ np.full(values.shape[1], 1.0 / values.shape[1])
