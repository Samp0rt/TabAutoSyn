"""
All distance computations use precomputed Gower distance matrices
instead of Euclidean distances on scaled arrays.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Optional
from dataclasses import dataclass

from scoring import gower_matrix


@dataclass
class Experiment1Metrics:
    prdc_precision: float
    prdc_recall: float

    nn_identifiability: float
    coverage_score: float          # bin-based distributional coverage (higher = better)
    mmd: float = 0.0               # Maximum Mean Discrepancy (lower = better)

    def to_dict(self) -> dict:
        return {
            "prdc_precision": round(self.prdc_precision, 4),
            "prdc_recall": round(self.prdc_recall, 4),
            "nn_identifiability": round(self.nn_identifiability, 4),
            "coverage_score": round(self.coverage_score, 4),
            "mmd": round(self.mmd, 4)
        }


def _gower_cross(
    df_ref: pd.DataFrame,
    df_query: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
) -> np.ndarray:
    """
    Compute (N_query, N_ref) cross Gower distance matrix.
    Each entry [i, j] = Gower distance between query row i and ref row j.
    """
    n_features = len(num_cols) + len(cat_cols)
    N_q = len(df_query)
    N_r = len(df_ref)
    D = np.zeros((N_q, N_r), dtype=np.float64)

    for col in num_cols:
        q_vals = df_query[col].fillna(0).values.astype(np.float64)
        r_vals = df_ref[col].fillna(0).values.astype(np.float64)
        # global range from ref (real data defines the scale)
        R = r_vals.max() - r_vals.min()
        if R < 1e-12:
            continue
        D += np.abs(q_vals[:, None] - r_vals[None, :]) / R

    for col in cat_cols:
        q_vals = np.array(df_query[col].fillna("__missing__").astype(str).tolist())
        r_vals = np.array(df_ref[col].fillna("__missing__").astype(str).tolist())
        D += (q_vals[:, None] != r_vals[None, :]).astype(np.float64)

    D /= n_features
    return D   # shape (N_query, N_ref)


def _kth_nn_dist_from_matrix(D_square: np.ndarray, k: int) -> np.ndarray:
    """
    Return k-th NN distance for each row of a SQUARE distance matrix.
    Index 0 = self-distance = 0, so k-th NN is at column index k.
    """
    sorted_d = np.sort(D_square, axis=1)
    return sorted_d[:, min(k, D_square.shape[1] - 1)]



def compute_prdc_gower(
    D_real_sq: np.ndarray,
    D_syn_sq: np.ndarray,
    D_syn_to_real: np.ndarray,
    D_real_to_syn: np.ndarray,
    k: int = 5,
) -> tuple[float, float]:
    """
    PRDC Precision and Recall using precomputed Gower distances.

    Precision: fraction of synthetic points within the real manifold
               i.e. d(s, nearest real) <= k-th NN radius of that real point
    Recall:    fraction of real points covered by the synthetic manifold
               i.e. d(r, nearest syn) <= k-th NN radius of that syn point

    Each manifold's k-NN radius is defined by its own internal distances,
    per the original PRDC paper. gower_matrix() normalises R by the passed
    dataframe, so D_real_sq uses real ranges and D_syn_sq uses syn ranges.
    In practice the ranges are similar (mmd is low), so any scale difference
    is negligible — and this is the semantically correct formulation.
    """
    real_radii = _kth_nn_dist_from_matrix(D_real_sq, k)   # (N_real,)
    syn_radii  = _kth_nn_dist_from_matrix(D_syn_sq,  k)   # (N_syn,)

    # Precision: syn point is in the real manifold if its nearest real
    # neighbour's k-NN radius covers it.
    nn_idx_syn   = D_syn_to_real.argmin(axis=1)
    dist_syn_min = D_syn_to_real.min(axis=1)
    precision    = float(np.mean(dist_syn_min <= real_radii[nn_idx_syn]))

    # Recall: real point is covered if the nearest syn point's k-NN radius
    # (measured among syn points) covers it.
    nn_idx_real   = D_real_to_syn.argmin(axis=1)
    dist_real_min = D_real_to_syn.min(axis=1)
    recall        = float(np.mean(dist_real_min <= syn_radii[nn_idx_real]))

    return precision, recall


def compute_coverage_gower(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    num_cols: list,
    n_bins: int = 10,
) -> float:
    """
    Distributional coverage score: fraction of quantile bins occupied by
    both real and synthetic data across all numeric features.

    For each numeric feature f:
        - Divide real data into n_bins quantile bins
        - Clip synthetic values to real [min, max] before binning so that
          out-of-range synthetic values don't get assigned to the boundary
          bins and inflate coverage artificially
        - Count bins that contain at least one (clipped) synthetic point
        - coverage_f = (occupied bins) / (non-empty real bins)

    Final score = mean over all numeric features.

    Clipping is intentional: a synthetic value outside the real range does
    NOT count as covering any real bin — it is simply out-of-distribution.
    """
    if not num_cols or len(df_syn) == 0:
        return 0.0

    scores = []
    for col in num_cols:
        real_vals = df_real[col].dropna().values.astype(float)
        syn_vals  = df_syn[col].apply(pd.to_numeric, errors="coerce").dropna().values

        if len(real_vals) == 0 or len(syn_vals) == 0:
            continue

        # quantile bin edges from real data
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.unique(np.percentile(real_vals, quantiles))

        if len(bin_edges) < 2:
            continue

        # clip synthetic values to real [min, max] — values outside this
        # range are out-of-distribution and must not inflate coverage
        real_min, real_max = bin_edges[0], bin_edges[-1]
        syn_vals_clipped = syn_vals[(syn_vals >= real_min) & (syn_vals <= real_max)]

        real_bin_ids = np.digitize(real_vals,        bin_edges[1:-1])
        syn_bin_ids  = np.digitize(syn_vals_clipped, bin_edges[1:-1])

        real_bins = set(real_bin_ids)
        syn_bins  = set(syn_bin_ids)
        occupied  = len(real_bins & syn_bins)

        scores.append(occupied / len(real_bins))

    return float(np.mean(scores)) if scores else 0.0


def compute_mmd(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    num_cols: list,
    gamma: float = None,
) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel on numeric features.
    Lower = distributions are more similar.

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    where k(a,b) = exp(-gamma * ||a-b||^2)
    gamma defaults to 1 / (2 * median_pairwise_dist^2)  (median heuristic)
    """
    from sklearn.preprocessing import StandardScaler

    common = [c for c in num_cols if c in df_syn.columns]
    if not common:
        return float("nan")

    X = df_real[common].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
    Y = df_syn[common].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)

    # scale to make gamma meaningful across datasets
    sc = StandardScaler()
    X = sc.fit_transform(X)
    Y = sc.transform(Y)

    # subsample for speed if large
    rng = np.random.default_rng(42)
    max_n = 500
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    if gamma is None:
        # median heuristic on combined sample
        Z = np.vstack([X, Y])
        dists = np.linalg.norm(Z[:, None] - Z[None, :], axis=2)
        median_d = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * median_d ** 2 + 1e-12)

    def rbf(A, B):
        dists_sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=2)
        return np.exp(-gamma * dists_sq)

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    n, m = len(X), len(Y)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) \
         - 2 * Kxy.mean()

    return float(max(mmd2, 0.0))  # clip numerical negatives



def evaluate_generation(
    df_syn: pd.DataFrame,
    df_real_test: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    #target_col: str,
    #task_type: str,
    #strategy: str,
    #n_examples: int,
    k_prdc: int = 5,
) -> Experiment1Metrics:
    """
    Experiment 1 evaluation.

    All distributional metrics (PRDC, MMD, coverage, nn_identifiability)
    are computed against df_real_TEST — the held-out split that was never
    seen by the LLM during generation.

    Rationale:
        df_syn was generated FROM df_selected ⊂ df_train.
        Comparing df_syn to df_train measures memorisation, not generalisation.
        Comparing df_syn to df_real_test measures whether the LLM captured
        the true data distribution beyond the examples it was shown.

    downstream_auc: XGBoost trained on df_syn, evaluated on df_real_test.
        This is unchanged — it always used df_real_test as the test set.

    Args:
        df_syn       : synthetic DataFrame from GEN.run()
        df_real_test : held-out real data, never seen by the LLM
        num_cols     : numeric feature column names
        cat_cols     : categorical feature column names
        target_col   : target column name
        task_type    : 'classification' or 'regression'
        strategy     : strategy name (for labelling)
        n_examples   : number of rows selected (for labelling)
        k_prdc       : neighbourhood size for PRDC
    """
    # align synthetic columns to test set
    feature_cols = num_cols + cat_cols
    for col in feature_cols:
        if col not in df_syn.columns:
            df_syn = df_syn.copy()
            df_syn[col] = np.nan

    # ── distance matrices against df_real_TEST ────────────────────────
    D_test_sq     = gower_matrix(df_real_test, num_cols, cat_cols)  # (N_test, N_test)
    D_syn_sq      = gower_matrix(df_syn,       num_cols, cat_cols)  # (N_syn,  N_syn)

    # (N_syn, N_test): syn rows → test rows  — R taken from real data
    D_syn_to_test = _gower_cross(df_real_test, df_syn, num_cols, cat_cols)
    D_test_to_syn = D_syn_to_test.T   # (N_test, N_syn)

    # ── PRDC vs test set ──────────────────────────────────────────────
    precision, recall = compute_prdc_gower(
        D_test_sq, D_syn_sq, D_syn_to_test, D_test_to_syn, k=k_prdc
    )

    # ── coverage vs test set ──────────────────────────────────────────
    coverage = compute_coverage_gower(df_real_test, df_syn, num_cols)

    # ── MMD vs test set ───────────────────────────────────────────────
    mmd = compute_mmd(df_real_test, df_syn, num_cols)

    # ── nn_identifiability vs test set ────────────────────────────────
    # Normalised proximity score ∈ [0, 1].
    #
    # Reference scale: internal_nn_dists[i] = distance from real point i to
    # its nearest real neighbour — captures local density of the real manifold.
    #
    # For each synthetic point s we compute its NN distance to test set,
    # then find its percentile rank within internal_nn_dists.
    # Final score = mean percentile rank across all synthetic points.
    #
    # This is strictly correct: every synthetic point is ranked individually,
    # so a generator that produces half "too-close" and half "too-far" points
    # will score ~0.5 on average but with high variance — which is detectable
    # separately. Collapsing to a single mean before ranking (previous version)
    # would silently average out those extremes.
    #
    # Complexity: O(N_test log N_test + N_syn log N_test) — fully vectorised.
    internal_nn_dists = np.sort(D_test_sq, axis=1)[:, 1]    # (N_test,)
    syn_nn_dists      = D_syn_to_test.min(axis=1)            # (N_syn,)
    sorted_internal   = np.sort(internal_nn_dists)
    # searchsorted gives the number of internal distances < each syn distance
    nn_id = float(
        np.searchsorted(sorted_internal, syn_nn_dists, side="right").mean()
        / len(sorted_internal)
    )


    return Experiment1Metrics(
        prdc_precision=precision,
        prdc_recall=recall,
        nn_identifiability=nn_id,
        coverage_score=coverage,
        mmd=mmd
    )
