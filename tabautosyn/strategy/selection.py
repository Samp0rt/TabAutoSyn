"""
Prompt example selection strategies (S1–S6).

All strategies operate on a precomputed Gower distance matrix, which
supports mixed numeric + categorical feature spaces natively.

Strategies
----------
S1_Random        : Uniform random sample — baseline.
S2_Center        : Densest examples (lowest nearest-neighbour privacy score).
S3_Tail          : Rarest examples (highest statistical isolation score).
S4_MaxCoverage   : Greedy farthest-point sampling for maximum spread.
S5_PrivacyAware  : Composite Pareto ranking of utility vs. privacy.
S6_Stratified    : K-means clustering on a classical MDS embedding;
                   one medoid per cluster.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from typing import Dict, List, Optional

from strategies.scoring import ExampleScores, pareto_score


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

STRATEGY_NAMES: List[str] = [
    "S1_Random",
    "S2_Center",
    "S3_Tail",
    "S4_MaxCoverage",
    "S5_PrivacyAware",
    "S6_Stratified",
]

_DEFAULT_ALPHAS: List[float] = [0.3, 0.5, 0.7]
_MDS_COMPONENTS: int = 10
_KMEANS_N_INIT: int = 10
_SVD_N_ITER: int = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _double_center(D_sq: np.ndarray) -> np.ndarray:
    """Return the doubly-centred Gram matrix B from a squared distance matrix.

    Applies the standard formula:
        B = -0.5 * (D² - row_mean - col_mean + grand_mean)

    This is equivalent to computing  B = -0.5 * H @ D² @ H
    where H = I - (1/N) * 11ᵀ is the centering matrix.
    """
    row_mean = D_sq.mean(axis=1, keepdims=True)
    col_mean = D_sq.mean(axis=0, keepdims=True)
    grand_mean = D_sq.mean()
    return -0.5 * (D_sq - row_mean - col_mean + grand_mean)


def _classical_mds(D: np.ndarray, n_components: int = _MDS_COMPONENTS) -> np.ndarray:
    """Embed N points into Euclidean space via Classical MDS (PCoA).

    Uses double centering followed by a randomised truncated SVD.
    Chosen over ``sklearn.manifold.MDS`` because it is:

    * **Deterministic** — no iterative optimisation or random restarts.
    * **Warning-free** — avoids ``FutureWarning`` from sklearn MDS kwargs.
    * **Safe on large datasets** — single-threaded; no ``n_jobs=-1`` SIGKILL.

    Negative eigenvalues caused by non-Euclidean Gower distances are clipped
    to zero before taking the square root.

    Parameters
    ----------
    D:
        (N, N) symmetric distance matrix.
    n_components:
        Number of embedding dimensions. Capped at N - 1.

    Returns
    -------
    np.ndarray
        (N, n_components) Euclidean embedding.
    """
    N = D.shape[0]
    n_components = min(n_components, N - 1)

    B = _double_center(D ** 2)

    # Randomised SVD is equivalent to eigen-decomposition for symmetric PSD B.
    U, singular_values, _ = randomized_svd(
        B,
        n_components=n_components,
        random_state=0,
        n_iter=_SVD_N_ITER,
    )
    # Clip small negatives introduced by numerical noise.
    eigenvalues = np.maximum(singular_values, 0.0)
    embedding = U * np.sqrt(eigenvalues)[np.newaxis, :]  # shape: (N, n_components)
    return embedding


def _farthest_point_sample(D: np.ndarray, n: int, rng: np.random.Generator) -> List[int]:
    """Greedy farthest-point sampling over a precomputed distance matrix.

    Iteratively adds the point that maximises the minimum distance to the
    current selection set, yielding a well-spread subset.

    Parameters
    ----------
    D:
        (N, N) precomputed distance matrix.
    n:
        Number of points to select.
    rng:
        NumPy random Generator used to draw the initial seed point.

    Returns
    -------
    List[int]
        Indices of the n selected points.
    """
    N = D.shape[0]
    seed_idx = int(rng.integers(0, N))
    selected = [seed_idx]
    # min_dists[i] tracks the shortest distance from point i to any selected point.
    min_dists = D[seed_idx].copy()

    for _ in range(n - 1):
        # Mask already-selected indices so they cannot be re-chosen.
        candidate_dists = min_dists.copy()
        candidate_dists[selected] = -np.inf
        next_idx = int(np.argmax(candidate_dists))
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, D[next_idx])

    return selected


def _kmeans_medoids(embedding: np.ndarray, n_clusters: int, random_state: int) -> List[int]:
    """Cluster the embedding and return the medoid index of each cluster.

    For each cluster the point closest (in L2) to the cluster centroid is
    selected as the representative (approximate medoid).

    Parameters
    ----------
    embedding:
        (N, d) array of embedded points.
    n_clusters:
        Number of clusters / examples to select.
    random_state:
        Passed to ``KMeans`` for reproducibility.

    Returns
    -------
    List[int]
        One index per cluster (the approximate medoid).
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=_KMEANS_N_INIT)
    km.fit(embedding)

    medoids: List[int] = []
    for cluster_id in range(n_clusters):
        member_indices = np.where(km.labels_ == cluster_id)[0]
        member_embeddings = embedding[member_indices]
        centroid = km.cluster_centers_[cluster_id]
        distances_to_centroid = np.linalg.norm(member_embeddings - centroid, axis=1)
        medoids.append(int(member_indices[np.argmin(distances_to_centroid)]))

    return medoids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_examples(
    D: np.ndarray,
    scores: ExampleScores,
    n: int,
    strategy: str,
    alpha: float = 0.5,
    random_state: int = 42,
) -> List[int]:
    """Select *n* example indices from a dataset using the specified strategy.

    Parameters
    ----------
    D:
        (N, N) precomputed Gower distance matrix.
    scores:
        Per-example scores produced by ``scoring.compute_scores``.
    n:
        Number of examples to select.
    strategy:
        Name of the selection strategy; must be one of ``STRATEGY_NAMES``.
    alpha:
        Privacy/utility trade-off weight used only by ``S5_PrivacyAware``.
        Higher values favour utility; lower values favour privacy.
    random_state:
        Seed for all random operations, ensuring reproducibility.

    Returns
    -------
    List[int]
        Indices of the *n* selected examples.

    Raises
    ------
    ValueError
        If *n* exceeds the dataset size or *strategy* is not recognised.
    """
    N = D.shape[0]
    if n > N:
        raise ValueError(
            f"Requested n={n} examples but the dataset only contains N={N} rows."
        )

    rng = np.random.default_rng(random_state)

    if strategy == "S1_Random":
        # Uniform random sample — provides a reproducible baseline.
        return rng.choice(N, size=n, replace=False).tolist()

    if strategy == "S2_Center":
        # Select the n examples with the lowest nearest-neighbour privacy
        # score, i.e. the most densely packed / typical examples.
        return np.argsort(scores.s_priv_nn)[:n].tolist()

    if strategy == "S3_Tail":
        # Select the n rarest examples (highest statistical isolation score).
        return np.argsort(scores.s_stat)[::-1][:n].tolist()

    if strategy == "S4_MaxCoverage":
        # Greedy farthest-point sampling ensures maximum feature-space spread.
        return _farthest_point_sample(D, n, rng)

    if strategy == "S5_PrivacyAware":
        # Rank by a composite Pareto score that balances utility and privacy.
        composite_scores = pareto_score(scores, alpha=alpha)
        return np.argsort(composite_scores)[::-1][:n].tolist()

    if strategy == "S6_Stratified":
        # Embed via Classical MDS, cluster with K-means, then pick the
        # approximate medoid of each cluster as the representative example.
        embedding = _classical_mds(D, n_components=min(_MDS_COMPONENTS, N - 1))
        return _kmeans_medoids(embedding, n_clusters=n, random_state=random_state)

    raise ValueError(
        f"Unknown strategy '{strategy}'. "
        f"Expected one of: {STRATEGY_NAMES}."
    )
