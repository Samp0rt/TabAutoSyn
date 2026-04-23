import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExampleScores:
    s_stat: np.ndarray        # (N,) mean k-NN Gower distance
    s_priv_nn: np.ndarray     # (N,) nearest-neighbor Gower distance
    s_stat_norm: np.ndarray   # normalized to [0, 1]
    s_priv_norm: np.ndarray   # normalized to [0, 1]
    D: np.ndarray             # (N, N) full Gower distance matrix


def gower_matrix(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
) -> np.ndarray:
    """
    (N, N) Gower distance matrix for mixed numeric + categorical data.

    Numeric feature f:   |xi_f - xj_f| / R_f,  R_f = max - min
    Categorical feature f:  1[xi_f != xj_f]
    Result = mean over all features  →  values in [0, 1].

    Range normalisation for numerics is done inside this function.
    """
    N = len(df)
    n_features = len(num_cols) + len(cat_cols)
    if n_features == 0:
        raise ValueError("gower_matrix: no features provided")

    D = np.zeros((N, N), dtype=np.float64)

    for col in num_cols:
        vals = df[col].fillna(df[col].median()).values.astype(np.float64)
        R = vals.max() - vals.min()
        if R < 1e-12:
            continue
        D += np.abs(vals[:, None] - vals[None, :]) / R

    for col in cat_cols:
        vals = np.array(df[col].fillna("__missing__").astype(str).tolist())
        D += (vals[:, None] != vals[None, :]).astype(np.float64)

    D /= n_features
    np.fill_diagonal(D, 0.0)
    return D


def gower_cross(
    df_ref: pd.DataFrame,
    df_query: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
) -> np.ndarray:
    """
    (N_query, N_ref) cross Gower distance matrix.
    Range R_f computed from df_ref (real data defines the scale).
    """
    n_features = len(num_cols) + len(cat_cols)
    D = np.zeros((len(df_query), len(df_ref)), dtype=np.float64)

    for col in num_cols:
        r_vals = df_ref[col].fillna(df_ref[col].median()).values.astype(np.float64)
        q_vals = df_query[col].fillna(df_ref[col].median()).values.astype(np.float64)
        R = r_vals.max() - r_vals.min()
        if R < 1e-12:
            continue
        D += np.abs(q_vals[:, None] - r_vals[None, :]) / R

    for col in cat_cols:
        r_vals = np.array(df_ref[col].fillna("__missing__").astype(str).tolist())
        q_vals = np.array(df_query[col].fillna("__missing__").astype(str).tolist())
        D += (q_vals[:, None] != r_vals[None, :]).astype(np.float64)

    if n_features > 0:
        D /= n_features
    return D


def compute_scores(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    k: Optional[int] = None,
) -> ExampleScores:
 
    N = len(df)
    if k is None:
        k = max(5, int(np.sqrt(N)))
    k = min(k, N - 1)

    D = gower_matrix(df, num_cols, cat_cols)

    nn_k = NearestNeighbors(n_neighbors=k + 1, metric="precomputed", n_jobs=-1)
    nn_k.fit(D)
    dists_k, _ = nn_k.kneighbors(D)
    s_stat = dists_k[:, 1:].mean(axis=1)       # skip self (col 0, dist=0)

    nn_1 = NearestNeighbors(n_neighbors=2, metric="precomputed", n_jobs=-1)
    nn_1.fit(D)
    dists_1, _ = nn_1.kneighbors(D)
    s_priv_nn = dists_1[:, 1]

    def _norm(v):
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    return ExampleScores(
        s_stat=s_stat,
        s_priv_nn=s_priv_nn,
        s_stat_norm=_norm(s_stat),
        s_priv_norm=_norm(s_priv_nn),
        D=D,
    )


def pareto_score(scores: ExampleScores, alpha: float = 0.5) -> np.ndarray:
    """
    S5 composite:  alpha * s_stat_norm  -  (1-alpha) * s_priv_norm
    alpha=1 → pure coverage (S3), alpha=0 → pure privacy (S2)
    """
    return alpha * scores.s_stat_norm - (1 - alpha) * scores.s_priv_norm
