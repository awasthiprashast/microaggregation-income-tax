"""
Micro-aggregation partitioning algorithms.

All partitioners take an (n x d) matrix X in the distance space produced by
``features.FeatureEncoder`` and return an integer label per record
(labels are 0..G-1). Aggregation (replacing each record by its group's
centroid) is done separately by ``aggregate`` so that the *same* partition is
applied to every released attribute.

MDAV - Maximum Distance to Average Vector
-----------------------------------------
Domingo-Ferrer & Mateo-Sanz (2002); Domingo-Ferrer & Torra (2005).

Optimal multivariate micro-aggregation (minimise the within-group sum of
squares subject to every group having >= k members) is NP-hard
(Oganian & Domingo-Ferrer, 2001). MDAV is the standard fixed-size heuristic:

    R <- all records
    while |R| >= 3k:
        c  <- centroid(R)
        r  <- argmax_{x in R} ||x - c||        (most extreme record)
        s  <- argmax_{x in R} ||x - r||        (most extreme w.r.t. r)
        G_r <- r + its k-1 nearest neighbours in R ;  R <- R \\ G_r
        G_s <- s + its k-1 nearest neighbours in R ;  R <- R \\ G_s
    if |R| >= 2k:
        c <- centroid(R);  r <- farthest from c
        G_r <- r + its k-1 nearest neighbours;  R <- R \\ G_r
    G_last <- R                                 (k <= |G_last| <= 2k-1)

Intuition: groups are built "from the outside in". Extreme records are the
hardest to hide, so they are handled first while they still have the most
candidate neighbours; picking r and s at opposite ends of the cloud keeps the
two groups formed in each iteration far apart.

Complexity: every iteration does O(|R|) distance evaluations of cost O(d) plus
two linear-time k-selections, and removes 2k records, so there are about
n/(2k) iterations and the total time is

    sum_{i=0}^{n/(2k)} O((n - 2ki) d)  =  O(n^2 d / k)

with O(n d) memory. (The literature usually quotes O(n^2) for fixed d and k.)

Determinism: ties are broken by the lowest original row index, so the same
input always yields the same partition (no randomness is involved).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_inputs(X: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array (records x features).")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values; clean the data first.")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer (got {k!r}).")
    if X.shape[0] < k:
        raise ValueError(f"Need at least k={k} records, got {X.shape[0]}.")
    return X


def _sq_dist_to(points: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row of ``points`` to vector ``p``."""
    diff = points - p
    return np.einsum("ij,ij->i", diff, diff)


def _k_nearest(sq_dist: np.ndarray, k: int, anchor: int) -> np.ndarray:
    """
    Positions of the k smallest distances, always including ``anchor``.

    Linear time (np.partition). Ties at the k-th distance are broken by the
    lowest position, which makes the selection deterministic.
    """
    m = sq_dist.size
    if k >= m:
        return np.arange(m)
    d = sq_dist.copy()
    d[anchor] = -np.inf                       # the anchor is always its own member
    kth = np.partition(d, k - 1)[k - 1]
    strictly_less = np.flatnonzero(d < kth)
    ties = np.flatnonzero(d == kth)
    return np.concatenate([strictly_less, ties[: k - strictly_less.size]])


def mdav(X: np.ndarray, k: int) -> np.ndarray:
    """
    Fixed-size MDAV partition.

    Returns
    -------
    labels : (n,) int array, group id of every record. All groups have exactly
             k records except the last one, which has between k and 2k-1.
    """
    X = _check_inputs(X, k)
    n = X.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    remaining = np.arange(n)                  # kept in increasing index order
    group = 0

    def take(positions: np.ndarray) -> None:
        nonlocal remaining, group
        labels[remaining[positions]] = group
        group += 1
        keep = np.ones(remaining.size, dtype=bool)
        keep[positions] = False
        remaining = remaining[keep]

    while remaining.size >= 3 * k:
        sub = X[remaining]
        centroid = sub.mean(axis=0)
        r = int(np.argmax(_sq_dist_to(sub, centroid)))
        d_r = _sq_dist_to(sub, sub[r])
        g_r = _k_nearest(d_r, k, anchor=r)

        # s = record farthest from r among those not just grouped with r.
        d_r_masked = d_r.copy()
        d_r_masked[g_r] = -np.inf
        s_global = remaining[int(np.argmax(d_r_masked))]
        take(g_r)

        sub = X[remaining]
        s = int(np.searchsorted(remaining, s_global))
        d_s = _sq_dist_to(sub, sub[s])
        take(_k_nearest(d_s, k, anchor=s))

    if remaining.size >= 2 * k:
        sub = X[remaining]
        r = int(np.argmax(_sq_dist_to(sub, sub.mean(axis=0))))
        take(_k_nearest(_sq_dist_to(sub, sub[r]), k, anchor=r))

    if remaining.size:
        take(np.arange(remaining.size))

    return labels


def random_partition(n: int, k: int, seed: int) -> np.ndarray:
    """
    Baseline: groups of size k formed uniformly at random (last group absorbs
    the remainder). Gives the same k-anonymity guarantee as MDAV but ignores
    similarity, so it shows how much utility the clustering step buys.
    """
    if n < k:
        raise ValueError(f"Need at least k={k} records, got {n}.")
    order = np.random.default_rng(seed).permutation(n)
    labels = np.empty(n, dtype=np.int64)
    n_groups = n // k
    group_of_position = np.minimum(np.arange(n) // k, n_groups - 1)
    labels[order] = group_of_position
    return labels


def univariate_groups(values: pd.Series, k: int) -> np.ndarray:
    """
    Individual-ranking groups for ONE attribute: sort the values (stable, ties
    by row order) and cut the ranking into consecutive blocks of k, the last
    block absorbing the remainder. This is the optimal-style univariate
    heuristic; it is O(n log n).
    """
    n = len(values)
    if n < k:
        raise ValueError(f"Need at least k={k} records, got {n}.")
    if pd.api.types.is_numeric_dtype(values):
        order = np.argsort(values.to_numpy(dtype=float), kind="stable")
    else:
        order = np.argsort(values.astype(str).to_numpy(), kind="stable")
    n_groups = n // k
    labels = np.empty(n, dtype=np.int64)
    labels[order] = np.minimum(np.arange(n) // k, n_groups - 1)
    return labels


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _group_mode(values: pd.Series, labels: np.ndarray) -> np.ndarray:
    """Most frequent level per group; ties -> alphabetically first level."""
    counts = pd.crosstab(labels, values.astype(str).to_numpy())
    counts = counts.reindex(sorted(counts.columns), axis=1)
    modes = counts.idxmax(axis=1)             # first max column == alphabetical tie-break
    return modes.reindex(range(labels.max() + 1)).to_numpy(dtype=object)[labels]


def aggregate(df: pd.DataFrame, labels: np.ndarray, numeric, categorical) -> pd.DataFrame:
    """
    Replace every record by its group centroid: arithmetic mean for numeric
    attributes (preserves totals and linear relations), mode for categorical
    attributes. Returns a frame aligned with ``df`` (same row order).
    """
    labels = np.asarray(labels)
    out = pd.DataFrame(index=df.index)
    for c in numeric:
        means = df[c].groupby(labels).mean()
        out[c] = means.to_numpy(dtype=float)[labels]
    for c in categorical:
        out[c] = _group_mode(df[c], labels)
    return out


def aggregate_individual_ranking(df: pd.DataFrame, k: int, numeric, categorical) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Individual ranking: a *separate* univariate micro-aggregation per attribute.
    Returns the aggregated frame and the per-attribute labels.
    """
    out = pd.DataFrame(index=df.index)
    per_attribute: dict[str, np.ndarray] = {}
    for c in numeric:
        lab = univariate_groups(df[c], k)
        per_attribute[c] = lab
        out[c] = df[c].groupby(lab).mean().to_numpy(dtype=float)[lab]
    for c in categorical:
        lab = univariate_groups(df[c], k)
        per_attribute[c] = lab
        out[c] = _group_mode(df[c], lab)
    return out, per_attribute


def group_sizes(labels: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(labels))
