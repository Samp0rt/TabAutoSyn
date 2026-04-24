"""
Synthetic Data Quality Evaluator
---------------------------------
Identifies real samples that are poorly reproduced by a synthetic dataset.
Uses three complementary methods:
  - Mahalanobis distance  : measures how far a real sample is from the synthetic distribution
  - K-Nearest Neighbors   : measures proximity of real samples to their closest synthetic neighbours
  - Density-based scoring : uses log-likelihood under the synthetic distribution as a quality proxy

All methods work in a StandardScaler-normalised feature space so that
features with different magnitudes are treated equally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _regularised_inverse(matrix: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Return the inverse of *matrix*, adding a small ridge to ensure invertibility."""
    regularised = matrix + np.eye(matrix.shape[0]) * reg
    return np.linalg.inv(regularised)


def _mahalanobis_distances(
    samples: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis distance from each row in *samples* to *mean*."""
    return np.array([mahalanobis(row, mean, cov_inv) for row in samples])


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class SyntheticQualityEvaluator:
    """Evaluate how well a synthetic dataset reproduces each sample in the real dataset.

    Parameters
    ----------
    df_real:
        Reference (real) dataset.
    df_syn:
        Synthetic dataset. Must contain at least the same columns as *df_real*;
        any extra columns are dropped automatically.
    """

    def __init__(self, df_real: pd.DataFrame, df_syn: pd.DataFrame) -> None:
        self.df_real = df_real.copy()
        self.df_syn = df_syn[df_real.columns].copy()

        if list(self.df_real.columns) != list(self.df_syn.columns):
            raise ValueError("Column mismatch between real and synthetic DataFrames.")

        self.features: list[str] = df_real.columns.tolist()
        self._real_scaled: np.ndarray | None = None
        self._syn_scaled: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def preprocess_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Scale both datasets using a StandardScaler fitted on the real data.

        The scaler is fitted on *real* data only so that the synthetic data is
        evaluated relative to the real distribution — not its own statistics.

        Returns
        -------
        real_scaled, syn_scaled : np.ndarray
        """
        scaler = StandardScaler()
        self._real_scaled = scaler.fit_transform(self.df_real)
        self._syn_scaled = scaler.transform(self.df_syn)
        return self._real_scaled, self._syn_scaled

    def _get_scaled(self) -> tuple[np.ndarray, np.ndarray]:
        """Return cached scaled arrays, computing them on first access."""
        if self._real_scaled is None:
            self.preprocess_data()
        return self._real_scaled, self._syn_scaled  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Method 1 — Mahalanobis distance
    # ------------------------------------------------------------------

    def mahalanobis_distance_method(
        self,
        robust: bool = False,
        percentile: float = 95,
    ) -> pd.DataFrame:
        """Identify poorly reproduced samples using Mahalanobis distance.

        Each real sample is compared against the synthetic distribution.
        Samples whose distance exceeds the *percentile* threshold are flagged.

        Parameters
        ----------
        robust:
            If *True*, use the Minimum Covariance Determinant estimator (more
            resistant to outliers in the synthetic data).
        percentile:
            Samples above this percentile of the distance distribution are flagged.

        Returns
        -------
        pd.DataFrame with columns:
            original_index, mahalanobis_distance, chi2_threshold, percentile_threshold
        """
        real_scaled, syn_scaled = self._get_scaled()

        # Estimate mean and covariance of the synthetic distribution
        if robust:
            estimator = MinCovDet(random_state=42).fit(syn_scaled)
            mean_syn = estimator.location_
            cov_syn = estimator.covariance_
        else:
            mean_syn = syn_scaled.mean(axis=0)
            cov_syn = np.cov(syn_scaled.T)

        cov_inv = _regularised_inverse(cov_syn)
        distances = _mahalanobis_distances(real_scaled, mean_syn, cov_inv)

        threshold = np.percentile(distances, percentile)
        chi2_threshold = chi2.ppf(percentile / 100, df=real_scaled.shape[1])
        mask = distances > threshold

        label = "robust" if robust else "standard"

        return (
            pd.DataFrame({
                "original_index": self.df_real.index[mask],
                "mahalanobis_distance": distances[mask],
                "chi2_threshold": chi2_threshold,
                "percentile_threshold": threshold,
            })
            .sort_values("mahalanobis_distance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Method 2 — K-Nearest Neighbours
    # ------------------------------------------------------------------

    def nearest_neighbor_method(
        self,
        n_neighbors: int = 5,
        percentile: float = 95,
    ) -> pd.DataFrame:
        """Identify poorly reproduced samples using KNN distance to the synthetic set.

        For every real sample the mean Euclidean distance to its *n_neighbors*
        nearest synthetic neighbours is computed. High distances indicate that
        the synthetic dataset has no similar samples nearby.

        Parameters
        ----------
        n_neighbors:
            Number of nearest neighbours to consider.
        percentile:
            Samples above this percentile of mean-distance values are flagged.

        Returns
        -------
        pd.DataFrame with columns: original_index, mean_knn_distance, threshold
        """
        real_scaled, syn_scaled = self._get_scaled()

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(syn_scaled)
        distances, _ = knn.kneighbors(real_scaled)

        mean_distances = distances.mean(axis=1)
        threshold = np.percentile(mean_distances, percentile)
        mask = mean_distances > threshold

        return (
            pd.DataFrame({
                "original_index": self.df_real.index[mask],
                "mean_knn_distance": mean_distances[mask],
                "threshold": threshold,
            })
            .sort_values("mean_knn_distance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Method 3 — Density-based log-likelihood
    # ------------------------------------------------------------------

    def density_based_method(self, percentile: float = 5) -> pd.DataFrame:
        """Identify poorly reproduced samples by their log-likelihood under the synthetic density.

        A Gaussian approximation of the synthetic distribution is used.
        Real samples with the *lowest* log-likelihood scores lie in regions
        that the synthetic data covers poorly.

        Parameters
        ----------
        percentile:
            Samples *below* this percentile of the log-likelihood distribution
            are flagged (lower = further from the synthetic density).

        Returns
        -------
        pd.DataFrame with columns: original_index, log_likelihood, threshold
        """
        real_scaled, syn_scaled = self._get_scaled()

        estimator = EmpiricalCovariance().fit(syn_scaled)
        mean_syn = syn_scaled.mean(axis=0)
        cov_inv = _regularised_inverse(estimator.covariance_)

        distances = _mahalanobis_distances(real_scaled, mean_syn, cov_inv)
        log_likelihood = -0.5 * distances ** 2

        threshold = np.percentile(log_likelihood, percentile)
        mask = log_likelihood < threshold

        return (
            pd.DataFrame({
                "original_index": self.df_real.index[mask],
                "log_likelihood": log_likelihood[mask],
                "threshold": threshold,
            })
            .sort_values("log_likelihood")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------------

    def combined_method(
        self,
        weights: dict[str, float] | None = None,
        percentile: float = 95,
    ) -> pd.DataFrame:
        """Identify poorly reproduced samples using a weighted composite score.

        The three individual scores (Mahalanobis, KNN, density) are each
        normalised to [0, 1] with MinMaxScaler and then combined linearly.
        Higher composite score → worse reproduction quality.

        Parameters
        ----------
        weights:
            Dictionary with keys ``'mahalanobis'``, ``'knn'``, ``'density'``
            and non-negative float values that sum to 1.
            Defaults to ``{'mahalanobis': 0.5, 'knn': 0.3, 'density': 0.2}``.
        percentile:
            Samples above this percentile of the combined score are flagged.

        Returns
        -------
        pd.DataFrame with columns:
            original_index, combined_score, mahalanobis_component,
            knn_component, density_component, threshold
        """
        if weights is None:
            weights = {"mahalanobis": 0.5, "knn": 0.3, "density": 0.2}

        real_scaled, syn_scaled = self._get_scaled()
        scaler = MinMaxScaler()

        # --- Mahalanobis component ---
        mean_syn = syn_scaled.mean(axis=0)
        cov_inv = _regularised_inverse(np.cov(syn_scaled.T))
        mahal = _mahalanobis_distances(real_scaled, mean_syn, cov_inv)
        mahal_norm = scaler.fit_transform(mahal.reshape(-1, 1)).ravel()

        # --- KNN component ---
        knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        knn.fit(syn_scaled)
        knn_distances, _ = knn.kneighbors(real_scaled)
        knn_norm = scaler.fit_transform(knn_distances.mean(axis=1).reshape(-1, 1)).ravel()

        # --- Density component (negated log-likelihood so that higher = worse) ---
        density_score = 0.5 * mahal ** 2  # equivalent to -log_likelihood
        density_norm = scaler.fit_transform(density_score.reshape(-1, 1)).ravel()

        # --- Weighted combination ---
        combined = (
            weights["mahalanobis"] * mahal_norm
            + weights["knn"] * knn_norm
            + weights["density"] * density_norm
        )

        threshold = np.percentile(combined, percentile)
        mask = combined > threshold

        return (
            pd.DataFrame({
                "original_index": self.df_real.index[mask],
                "combined_score": combined[mask],
                "mahalanobis_component": mahal_norm[mask],
                "knn_component": knn_norm[mask],
                "density_component": density_norm[mask],
                "threshold": threshold,
            })
            .sort_values("combined_score", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def select_poorly_reproduced_samples(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    weights: dict[str, float] | None = None,
    percentile: float = 90,
) -> pd.DataFrame:
    """Return real samples that are poorly reproduced by the synthetic dataset.

    Uses the combined scoring method of :class:`SyntheticQualityEvaluator`.

    Parameters
    ----------
    df_real:
        Reference (real) dataset.
    df_syn:
        Synthetic dataset.
    weights:
        Passed directly to :meth:`SyntheticQualityEvaluator.combined_method`.
    percentile:
        Samples above this combined-score percentile are considered poorly
        reproduced.

    Returns
    -------
    pd.DataFrame
        Subset of *df_real* containing only the poorly reproduced rows.
    """
    if weights is None:
        weights = {"mahalanobis": 0.5, "knn": 0.3, "density": 0.2}

    evaluator = SyntheticQualityEvaluator(df_real, df_syn)
    poor_combined = evaluator.combined_method(weights=weights, percentile=percentile)

    poor_indices = poor_combined["original_index"].values
    df_poor = df_real.loc[poor_indices]

    return df_poor