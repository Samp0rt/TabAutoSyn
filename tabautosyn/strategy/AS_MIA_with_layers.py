"""
MIA with Explicit Privacy Layer Modeling
=========================================

Concept
-------
Our audit has always been an "upper bound" — the auditor knows everything:
the selection strategy, df_selected, and the true data distribution.

A real attacker knows less. The less they know, the weaker their attack.
Each layer of ignorance acts as a privacy protection mechanism.

Three protection layers are modeled explicitly as three attack modes:

  LAYER_0 — Oracle (our audit, upper bound)
  ------------------------------------------
  The attacker knows:
    - exactly which rows were in the prompt (df_selected)
    - the true distribution of real data (df_holdout from the same slice)
    - propensity matching removes density bias
  This is impossible in practice. Gives the maximum achievable AUC.

  LAYER_1 — Knows df_train, does not know selection strategy
  -----------------------------------------------------------
  The attacker knows:
    - the full df_train (e.g., a leaked client list)
    - does NOT know which specific rows were included in the prompt
  Modeled as: members = full df_train.
  With a random selection strategy, most training rows appeared in the
  prompt infrequently — signal is diluted. AUC drops.

  LAYER_2 — Does not know distribution (blind attacker)
  ------------------------------------------------------
  The attacker knows:
    - only the synthetic data df_syn
    - some suspected rows (not necessarily from the training set)
  Does NOT know:
    - df_train, df_selected
    - the true distribution → cannot perform propensity matching
    - no in-domain holdout → uses a random holdout
  Modeled as:
    - matching disabled (random holdout, no alignment)
    - normalization disabled (no background distribution available)
    - members = df_train (does not know df_selected)
  This is the most realistic scenario for an external attacker.

Interpreting Results
--------------------
  layer0_auc >> layer2_auc  →  protection works: ignorance significantly
                                degrades attack effectiveness
  layer0_auc ≈ layer2_auc   →  no protection: the signal is strong enough
                                to survive without distribution knowledge

Privacy score: layer_privacy_score = 1 - (layer2_auc - 0.5) / (layer0_auc - 0.5 + 1e-8)
  → 1.0: attack fully degrades without distribution knowledge
  → 0.0: privacy layers provide no protection
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from scoring import gower_cross


# =============================================================================
#  Result dataclass
# =============================================================================

@dataclass
class MIAMetrics:
    """
    Full set of metrics for a single MIA evaluation run.

    Fields are grouped by:
      - experiment metadata
      - Attribution Score (few-shot only, optional)
      - MIA Layer 0: Oracle (upper bound)
      - MIA Layer 1: attacker knows df_train, not selection strategy
      - MIA Layer 2: blind attacker (realistic external threat)
      - aggregate privacy score
    """

    # --- Experiment metadata -------------------------------------------------
    strategy: str
    n_examples: int
    batch_size: int
    n_batches: int

    # --- Attribution Score (computed for few-shot only when enabled) ---------
    attribution_score: float
    attribution_score_null_mean: float
    attribution_score_null_std: float
    attribution_score_pvalue: float
    attribution_score_ci_low: float
    attribution_score_ci_high: float
    mean_dist_to_own_batch: float
    mean_dist_to_other_batches: float

    # --- MIA Layer 0: Oracle (upper bound) -----------------------------------
    mia_auc: float                  # alias for layer0_auc; primary metric
    mia_auc_null_mean: float
    mia_auc_null_std: float
    mia_auc_pvalue: float
    mia_auc_ci_low: float
    mia_auc_ci_high: float
    mia_tpr_at_fpr10: float
    mia_matching_max_smd: float

    # --- MIA Layer 1: attacker knows df_train, not selection strategy --------
    mia_layer1_auc: float
    mia_layer1_null_mean: float
    mia_layer1_pvalue: float

    # --- MIA Layer 2: blind attacker (realistic scenario) --------------------
    mia_layer2_auc: float
    mia_layer2_null_mean: float
    mia_layer2_pvalue: float

    # --- Aggregate privacy protection score ----------------------------------
    mia_privacy_score: float        # how much ignorance protects; 1.0=full, 0.0=none

    n_members: int = 0
    n_non_members: int = 0

    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize all metrics to a flat dictionary with rounded floats."""
        return {
            # Metadata
            "strategy":                     self.strategy,
            "n_examples":                   self.n_examples,
            "batch_size":                   self.batch_size,
            "n_batches":                    self.n_batches,
            # Attribution Score
            "attribution_score":            round(self.attribution_score, 4),
            "attribution_score_null_mean":  round(self.attribution_score_null_mean, 4),
            "attribution_score_null_std":   round(self.attribution_score_null_std, 4),
            "attribution_score_pvalue":     round(self.attribution_score_pvalue, 4),
            "attribution_score_ci_low":     round(self.attribution_score_ci_low, 4),
            "attribution_score_ci_high":    round(self.attribution_score_ci_high, 4),
            "attribution_lift":             round(
                self.attribution_score - self.attribution_score_null_mean, 4),
            "mean_dist_to_own_batch":       round(self.mean_dist_to_own_batch, 4),
            "mean_dist_to_other_batches":   round(self.mean_dist_to_other_batches, 4),
            # MIA Layer 0
            "mia_auc":                      round(self.mia_auc, 4),
            "mia_auc_null_mean":            round(self.mia_auc_null_mean, 4),
            "mia_auc_null_std":             round(self.mia_auc_null_std, 4),
            "mia_auc_pvalue":               round(self.mia_auc_pvalue, 4),
            "mia_auc_ci_low":               round(self.mia_auc_ci_low, 4),
            "mia_auc_ci_high":              round(self.mia_auc_ci_high, 4),
            "mia_lift":                     round(
                self.mia_auc - self.mia_auc_null_mean, 4),
            "mia_tpr_at_fpr10":             round(self.mia_tpr_at_fpr10, 4),
            "mia_matching_max_smd":         round(self.mia_matching_max_smd, 4),
            # MIA Layer 1
            "mia_layer1_auc":               round(self.mia_layer1_auc, 4),
            "mia_layer1_null_mean":         round(self.mia_layer1_null_mean, 4),
            "mia_layer1_pvalue":            round(self.mia_layer1_pvalue, 4),
            "mia_layer1_lift":              round(
                self.mia_layer1_auc - self.mia_layer1_null_mean, 4),
            # MIA Layer 2
            "mia_layer2_auc":               round(self.mia_layer2_auc, 4),
            "mia_layer2_null_mean":         round(self.mia_layer2_null_mean, 4),
            "mia_layer2_pvalue":            round(self.mia_layer2_pvalue, 4),
            "mia_layer2_lift":              round(
                self.mia_layer2_auc - self.mia_layer2_null_mean, 4),
            # Privacy
            "mia_privacy_score":            round(self.mia_privacy_score, 4),
            "n_members":                    self.n_members,
            "n_non_members":                self.n_non_members,
        }

    def is_significant(self, alpha: float = 0.05) -> Dict[str, bool]:
        """
        Returns significance flags for each metric group.

        Attribution Score and Layer 0 use a combined criterion:
        p-value below alpha AND the confidence interval excludes the null (0.5).
        Layers 1 and 2 use only p-value (no CI stored).
        """
        as_ci_excludes_null  = self.attribution_score_ci_low > 0.5
        mia_ci_excludes_null = self.mia_auc_ci_low > 0.5
        return {
            "attribution_score": (
                self.attribution_score_pvalue < alpha and as_ci_excludes_null
            ),
            "mia_auc_layer0": (
                self.mia_auc_pvalue < alpha and mia_ci_excludes_null
            ),
            "mia_auc_layer1": self.mia_layer1_pvalue < alpha,
            "mia_auc_layer2": self.mia_layer2_pvalue < alpha,
        }

    def privacy_summary(self) -> str:
        """Human-readable interpretation of the three MIA privacy layers."""
        gap_l0_l1 = self.mia_auc - self.mia_layer1_auc
        gap_l1_l2 = self.mia_layer1_auc - self.mia_layer2_auc

        if self.mia_layer2_auc < 0.6:
            verdict = "✓ Low real-world threat: blind attack is ineffective"
        elif self.mia_layer2_auc < 0.7:
            verdict = "⚠ Moderate threat: blind attack yields a weak signal"
        else:
            verdict = "✗ High threat: signal is detectable without distribution knowledge"

        lines = [
            "━━━ MIA Privacy Layer Analysis ━━━",
            f"  Layer 0 (oracle):             AUC={self.mia_auc:.3f}"
            f"  ← upper bound, auditor knows everything",
            f"  Layer 1 (knows df_train):     AUC={self.mia_layer1_auc:.3f}"
            f"  ← random strategy dilutes signal",
            f"  Layer 2 (blind):              AUC={self.mia_layer2_auc:.3f}"
            f"  ← realistic external attacker",
            f"  Privacy score:                {self.mia_privacy_score:.3f}"
            f"  ← 1.0=full protection, 0.0=none",
            f"  Gap L0→L1 (selection randomness):    -{gap_l0_l1:.3f}",
            f"  Gap L1→L2 (distribution ignorance):  -{gap_l1_l2:.3f}",
            f"  Verdict: {verdict}",
        ]
        return "\n".join(lines)


# =============================================================================
#  Low-level helpers
# =============================================================================

def _bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a given statistic."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    boot_stats = np.array([
        statistic(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - ci
    return (
        float(np.percentile(boot_stats, 100 * alpha / 2)),
        float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    )


def _permutation_ci(null_aucs: np.ndarray, ci: float = 0.95) -> Tuple[float, float]:
    """Confidence interval derived from a permutation null distribution."""
    alpha = 1.0 - ci
    return (
        float(np.percentile(null_aucs, 100 * alpha / 2)),
        float(np.percentile(null_aucs, 100 * (1 - alpha / 2))),
    )


def _batch_slices(n_rows: int, batch_size: int) -> List[slice]:
    """Split n_rows into contiguous slices of batch_size."""
    return [
        slice(start, min(start + batch_size, n_rows))
        for start in range(0, n_rows, batch_size)
    ]


def _batch_assignment_array(batches: List[slice], n_syn: int) -> np.ndarray:
    """Map each synthetic row index to its batch index."""
    assignments = np.empty(n_syn, dtype=int)
    for batch_idx, sl in enumerate(batches):
        syn_start = sl.start
        syn_stop  = min(sl.stop, n_syn)
        if syn_start < syn_stop:
            assignments[syn_start:syn_stop] = batch_idx
    return assignments


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float = 0.1) -> float:
    """True Positive Rate at a given False Positive Rate threshold."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, fpr_threshold)
    return float(tpr[min(idx, len(tpr) - 1)])


# =============================================================================
#  Attribution Score
# =============================================================================

def _compute_as_from_assignments(
    D_cross: np.ndarray,
    batches: List[slice],
    syn_batch_assignments: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute Attribution Score from a (possibly permuted) batch assignment.

    Returns
    -------
    score : float
        Fraction of synthetic rows closer to their own batch than to others.
    mean_dist_own : float
        Mean distance from synthetic row to its own-batch members.
    mean_dist_other : float
        Mean distance from synthetic row to other-batch members.
    """
    n_syn   = D_cross.shape[0]
    n_real  = D_cross.shape[1]
    d_own   = np.full(n_syn, np.nan)
    d_other = np.full(n_syn, np.nan)

    for batch_idx, sl in enumerate(batches):
        own_cols = list(range(sl.start, sl.stop))
        if not own_cols:
            continue
        syn_rows = np.where(syn_batch_assignments == batch_idx)[0]
        if len(syn_rows) == 0:
            continue
        other_cols = [c for c in range(n_real) if c not in set(own_cols)]

        d_own[syn_rows] = D_cross[np.ix_(syn_rows, own_cols)].min(axis=1)
        if other_cols:
            d_other[syn_rows] = D_cross[np.ix_(syn_rows, other_cols)].min(axis=1)
        else:
            d_other[syn_rows] = D_cross[syn_rows, :].mean(axis=1)

    valid = ~(np.isnan(d_own) | np.isnan(d_other))
    if not valid.any():
        return 0.5, 0.0, 0.0

    delta = d_other[valid] - d_own[valid]
    return float(np.mean(delta > 0)), float(d_own[valid].mean()), float(d_other[valid].mean())


def compute_batch_attribution_score(
    df_selected: pd.DataFrame,
    df_syn: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    batch_size: int,
    rng: np.random.Generator,
    n_permutations: int = 1000,
) -> Tuple[float, float, float, float, float, float, float, float, List[slice]]:
    """
    Batch Attribution Score with permutation null distribution and bootstrap CI.

    Returns
    -------
    (score, null_mean, null_std, p_value, ci_low, ci_high,
     mean_dist_own, mean_dist_other, batches)
    """
    batches          = _batch_slices(len(df_selected), batch_size)
    n_syn            = len(df_syn)
    D_cross          = gower_cross(df_selected, df_syn, num_cols, cat_cols)
    true_assignments = _batch_assignment_array(batches, n_syn)

    score, mean_d_own, mean_d_other = _compute_as_from_assignments(
        D_cross, batches, true_assignments,
    )

    null_scores = np.array([
        _compute_as_from_assignments(D_cross, batches, rng.permutation(true_assignments))[0]
        for _ in range(n_permutations)
    ])

    null_mean = float(null_scores.mean())
    null_std  = float(null_scores.std())
    p_value   = float(np.mean(null_scores >= score))

    # Bootstrap CI on the per-row attribution indicator
    d_own_obs   = np.full(n_syn, np.nan)
    d_other_obs = np.full(n_syn, np.nan)

    for batch_idx, sl in enumerate(batches):
        own_cols = list(range(sl.start, sl.stop))
        syn_rows = np.where(true_assignments == batch_idx)[0]
        if not own_cols or len(syn_rows) == 0:
            continue
        other_cols = [c for c in range(D_cross.shape[1]) if c not in set(own_cols)]
        d_own_obs[syn_rows] = D_cross[np.ix_(syn_rows, own_cols)].min(axis=1)
        if other_cols:
            d_other_obs[syn_rows] = D_cross[np.ix_(syn_rows, other_cols)].min(axis=1)
        else:
            d_other_obs[syn_rows] = D_cross[syn_rows, :].mean(axis=1)

    valid      = ~(np.isnan(d_own_obs) | np.isnan(d_other_obs))
    indicators = (d_other_obs[valid] > d_own_obs[valid]).astype(float)
    ci_low, ci_high = _bootstrap_ci(indicators, statistic=np.mean, rng=rng)

    return score, null_mean, null_std, p_value, ci_low, ci_high, mean_d_own, mean_d_other, batches


# =============================================================================
#  MIA classifier and feature extraction
# =============================================================================

def _make_classifier() -> LogisticRegression:
    """Regularized logistic regression for MIA classification."""
    return LogisticRegression(C=0.01, max_iter=1000, random_state=42)


def _distance_features_raw(D: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Extract distance-based features from a distance matrix row set.

    Features per row: mean k-NN distance, min distance, std of k-NN,
    fraction of points within median radius.
    """
    k        = min(k, D.shape[1])
    sorted_d = np.sort(D, axis=1)[:, :k]
    f_mean   = sorted_d.mean(axis=1)
    f_min    = sorted_d[:, 0]
    f_std    = sorted_d.std(axis=1)
    radius   = float(np.median(f_min)) if len(f_min) > 0 else 1.0
    f_density = (D <= radius).mean(axis=1)
    return np.column_stack([f_mean, f_min, f_std, f_density])


def _distance_features_normalized(
    D_members: np.ndarray,
    D_non_members: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-score normalize distance features over the pooled member/non-member set.

    Normalization removes density bias introduced by propensity matching.
    Used in Layer 0 and Layer 1 (attacker knows the true distribution).
    """
    F_m   = _distance_features_raw(D_members)
    F_nm  = _distance_features_raw(D_non_members)
    F_pool = np.vstack([F_m, F_nm])
    mu     = F_pool.mean(axis=0)
    sigma  = F_pool.std(axis=0) + 1e-8
    return (F_m - mu) / sigma, (F_nm - mu) / sigma


def _compute_mia_auc(
    F_members: np.ndarray,
    F_non_members: np.ndarray,
    rng: np.random.Generator,
    n_splits: int,
) -> float:
    """Cross-validated AUC of the MIA classifier."""
    F = np.vstack([F_members, F_non_members])
    y = np.array([1] * len(F_members) + [0] * len(F_non_members))
    if len(y) < 4 or len(np.unique(y)) < 2:
        return 0.5

    n_splits   = max(2, min(n_splits, min(len(F_members), len(F_non_members))))
    n_repeats  = 20 if len(y) < 30 else 10
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=int(rng.integers(1 << 31)),
    )
    try:
        return float(cross_val_score(_make_classifier(), F, y, cv=cv, scoring="roc_auc").mean())
    except ValueError:
        return 0.5


def _run_mia_core(
    D_members: np.ndarray,
    D_non_members: np.ndarray,
    rng: np.random.Generator,
    n_null_runs: int,
    normalize: bool,
) -> Tuple[float, float, float, float, float, float]:
    """
    Core MIA evaluation: observed AUC + permutation null distribution.

    Parameters
    ----------
    normalize : bool
        True  → z-score normalization over the pool (Layer 0, Layer 1).
        False → raw distance features; no background distribution needed (Layer 2).

    Returns
    -------
    (auc, null_mean, null_std, p_value, ci_low, ci_high)
    """
    n_members     = len(D_members)
    n_non_members = len(D_non_members)
    n_all         = n_members + n_non_members
    n_splits      = max(2, min(5, min(n_members, n_non_members)))

    D_pool = np.vstack([D_members, D_non_members])

    def _extract_features(D_m: np.ndarray, D_nm: np.ndarray):
        if normalize:
            return _distance_features_normalized(D_m, D_nm)
        return _distance_features_raw(D_m), _distance_features_raw(D_nm)

    F_m_obs, F_nm_obs = _extract_features(D_members, D_non_members)
    observed_auc = _compute_mia_auc(F_m_obs, F_nm_obs, rng, n_splits=n_splits)

    row_indices = np.arange(n_all)
    null_aucs   = np.empty(n_null_runs)

    for i in range(n_null_runs):
        shuffled  = rng.permutation(row_indices)
        D_m_null  = D_pool[shuffled[:n_members]]
        D_nm_null = D_pool[shuffled[n_members:]]
        if len(D_m_null) == 0 or len(D_nm_null) == 0:
            null_aucs[i] = 0.5
            continue
        F_m_null, F_nm_null = _extract_features(D_m_null, D_nm_null)
        null_aucs[i] = _compute_mia_auc(F_m_null, F_nm_null, rng, n_splits=n_splits)

    null_mean = float(null_aucs.mean())
    null_std  = float(null_aucs.std())
    p_value   = float(np.mean(null_aucs >= observed_auc))
    ci_low, ci_high = _permutation_ci(null_aucs)

    return observed_auc, null_mean, null_std, p_value, ci_low, ci_high


# =============================================================================
#  Matching helpers
# =============================================================================

def _compute_smd(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    num_cols: List[str],
) -> Dict[str, float]:
    """Standardized Mean Difference (SMD) for numeric columns between two groups."""
    smds = {}
    for col in num_cols:
        a = df_a[col].dropna().values
        b = df_b[col].dropna().values
        if len(a) == 0 or len(b) == 0:
            smds[col] = np.nan
            continue
        pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        smds[col]  = abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
    return smds


def _propensity_match(
    df_members: pd.DataFrame,
    df_holdout: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    n_pick: int,
    rng: np.random.Generator,
    smd_warn_threshold: float = 0.1,
) -> Tuple[pd.DataFrame, float]:
    """
    Select non-members from df_holdout via propensity score matching.

    Fits a logistic regression to predict membership, then picks holdout rows
    with propensity scores closest to 0.5 (most uncertain → best match).
    Falls back to 1-NN Gower matching if logistic regression fails.

    Returns
    -------
    df_matched : pd.DataFrame
        Selected non-member rows.
    max_smd : float
        Maximum SMD across numeric columns after matching.
    """
    n_pick = min(n_pick, len(df_holdout))
    if n_pick == 0:
        return df_holdout.iloc[:0].reset_index(drop=True), 0.0

    try:
        df_all = pd.concat(
            [df_members.assign(_is_member=1),
             df_holdout.assign(_is_member=0)],
            ignore_index=True,
        )
        X_enc = pd.get_dummies(
            df_all[num_cols + cat_cols], columns=cat_cols, drop_first=True,
        ).fillna(0.0)
        y = df_all["_is_member"].values

        if len(np.unique(y)) < 2 or len(df_members) < 2 or len(df_holdout) < 2:
            raise ValueError("Too few samples for propensity model.")

        sc  = StandardScaler()
        X_s = sc.fit_transform(X_enc.values)
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_s, y)

        # Select holdout rows with propensity closest to 0.5
        ps           = clf.predict_proba(X_s[len(df_members):])[:, 1]
        dist_to_half = np.abs(ps - 0.5)
        top_idx      = np.argpartition(dist_to_half, n_pick - 1)[:n_pick]
        df_matched   = df_holdout.iloc[top_idx].reset_index(drop=True)

    except Exception as exc:
        warnings.warn(
            f"Propensity matching failed ({exc}), falling back to 1-NN.",
            RuntimeWarning,
        )
        D_hm = gower_cross(df_holdout, df_members, num_cols, cat_cols)
        if D_hm.shape == (len(df_members), len(df_holdout)):
            D_hm = D_hm.T
        top_idx    = np.argpartition(D_hm.min(axis=1), n_pick - 1)[:n_pick]
        df_matched = df_holdout.iloc[top_idx].reset_index(drop=True)

    smd_dict = _compute_smd(df_members, df_matched, num_cols)
    max_smd  = max((v for v in smd_dict.values() if not np.isnan(v)), default=0.0)

    if max_smd > smd_warn_threshold:
        bad_cols = [c for c, v in smd_dict.items() if not np.isnan(v) and v > smd_warn_threshold]
        warnings.warn(
            f"Layer 0 matching: max SMD={max_smd:.3f} > {smd_warn_threshold} "
            f"for columns {bad_cols}. AUC may be slightly inflated.",
            UserWarning,
        )
    return df_matched, max_smd


def _random_match(
    df_holdout: pd.DataFrame,
    n_pick: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Select non-members from df_holdout at random (no alignment).

    Used in Layer 2 to model an attacker who does not know the true
    data distribution and therefore cannot perform meaningful matching.
    """
    n_pick = min(n_pick, len(df_holdout))
    idx    = rng.choice(len(df_holdout), size=n_pick, replace=False)
    return df_holdout.iloc[idx].reset_index(drop=True)


# =============================================================================
#  Distance matrix builder (shared across all MIA layers)
# =============================================================================

def _build_distance_matrices(
    df_members: pd.DataFrame,
    df_non_members: pd.DataFrame,
    df_syn_aligned: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gower distance matrices from members and non-members to synthetic data.

    Returns
    -------
    D_members     : shape (n_members, n_syn)
    D_non_members : shape (n_non_members, n_syn)
    """
    n_members     = len(df_members)
    n_non_members = len(df_non_members)
    n_syn         = len(df_syn_aligned)

    df_pool = pd.concat(
        [df_members.reset_index(drop=True), df_non_members.reset_index(drop=True)],
        axis=0, ignore_index=True,
    )
    D_pool = gower_cross(df_pool, df_syn_aligned, num_cols, cat_cols)
    if D_pool.shape == (n_syn, n_members + n_non_members):
        D_pool = D_pool.T
    assert D_pool.shape == (n_members + n_non_members, n_syn)

    return D_pool[:n_members], D_pool[n_members:]


# =============================================================================
#  MIA layers
# =============================================================================

def compute_mia_layer0_oracle(
    df_selected: pd.DataFrame,
    df_syn_aligned: pd.DataFrame,
    df_holdout: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    rng: np.random.Generator,
    n_null_runs: int,
    max_non_members: Optional[int],
) -> dict:
    """
    LAYER 0 — Oracle (upper bound).

    The auditor knows exactly which rows were in the prompt (df_selected).
    Non-members are selected via propensity matching to remove density bias.
    Features are z-score normalized over the pool.

    This is the maximum achievable AUC; a real attacker cannot exceed it.
    """
    n_members = len(df_selected)
    n_pick    = min(len(df_holdout), max_non_members or n_members * 3)

    df_non_members, max_smd = _propensity_match(
        df_members=df_selected,
        df_holdout=df_holdout,
        num_cols=num_cols,
        cat_cols=cat_cols,
        n_pick=n_pick,
        rng=rng,
    )
    D_m, D_nm = _build_distance_matrices(
        df_selected, df_non_members, df_syn_aligned, num_cols, cat_cols,
    )
    auc, null_mean, null_std, pvalue, ci_low, ci_high = _run_mia_core(
        D_m, D_nm, rng, n_null_runs, normalize=True,
    )
    tpr_at_fpr10 = _compute_tpr_fpr10(D_m, D_nm, rng)

    return dict(
        auc=auc, null_mean=null_mean, null_std=null_std,
        pvalue=pvalue, ci_low=ci_low, ci_high=ci_high,
        tpr_at_fpr10=tpr_at_fpr10, max_smd=max_smd,
        n_members=len(D_m), n_non_members=len(D_nm),
    )


def compute_mia_layer1_knows_train(
    df_train: pd.DataFrame,
    df_syn_aligned: pd.DataFrame,
    df_holdout: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    rng: np.random.Generator,
    n_null_runs: int,
    max_non_members: Optional[int],
) -> dict:
    """
    LAYER 1 — Attacker knows df_train, not selection strategy.

    The attacker has access to the full training set (e.g., via a data leak)
    but does not know which specific rows were included in the prompt.

    Members = full df_train. With a random selection strategy, each training
    row appears in only a fraction of batches — signal is diluted across the
    entire training set, causing AUC to drop relative to Layer 0.

    Normalization is kept (attacker knows the distribution via df_train).
    Propensity matching is used since holdout comes from the same domain.
    """
    n_members = len(df_train)
    n_pick    = min(len(df_holdout), max_non_members or n_members * 3)

    df_non_members, _ = _propensity_match(
        df_members=df_train,
        df_holdout=df_holdout,
        num_cols=num_cols,
        cat_cols=cat_cols,
        n_pick=n_pick,
        rng=rng,
    )
    D_m, D_nm = _build_distance_matrices(
        df_train, df_non_members, df_syn_aligned, num_cols, cat_cols,
    )
    auc, null_mean, _, pvalue, _, _ = _run_mia_core(
        D_m, D_nm, rng, n_null_runs, normalize=True,
    )
    return dict(auc=auc, null_mean=null_mean, pvalue=pvalue)


def compute_mia_layer2_blind(
    df_train: pd.DataFrame,
    df_syn_aligned: pd.DataFrame,
    df_holdout: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    rng: np.random.Generator,
    n_null_runs: int,
    max_non_members: Optional[int],
) -> dict:
    """
    LAYER 2 — Blind attacker (most realistic external threat scenario).

    The attacker:
      - has access to df_train (or any suspected rows)
      - does NOT know the true data distribution → cannot use propensity matching
      - does NOT know the selection strategy
      - uses a random holdout without domain alignment

    Differences from Layer 0:
      1. members = df_train (unknown df_selected)
      2. non-members = random holdout (no matching)
      3. normalize=False: no z-score normalization (unknown background distribution)

    This represents the lower bound of real-world attacker capability.
    """
    n_members = len(df_train)
    n_pick    = min(len(df_holdout), max_non_members or n_members * 3)

    df_non_members = _random_match(df_holdout, n_pick, rng)
    D_m, D_nm = _build_distance_matrices(
        df_train, df_non_members, df_syn_aligned, num_cols, cat_cols,
    )
    # Raw features, no normalization: attacker has no knowledge of true density
    auc, null_mean, _, pvalue, _, _ = _run_mia_core(
        D_m, D_nm, rng, n_null_runs, normalize=False,
    )
    return dict(auc=auc, null_mean=null_mean, pvalue=pvalue)


def _compute_tpr_fpr10(
    D_members: np.ndarray,
    D_non_members: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """
    Compute TPR@FPR=0.1 using leave-one-out (small n) or stratified k-fold scores.

    Member scores are estimated out-of-sample to avoid optimistic bias.
    Non-member scores are obtained in-sample for the FPR axis.
    """
    n_m  = len(D_members)
    n_nm = len(D_non_members)
    if n_m < 2 or n_nm < 2:
        return 0.0

    F_m, F_nm = _distance_features_normalized(D_members, D_non_members)
    probas     = np.full(n_m, 0.5)

    use_loo = n_m <= 50
    if use_loo:
        for i in range(n_m):
            mask    = np.ones(n_m, dtype=bool)
            mask[i] = False
            F_tr = np.vstack([F_m[mask], F_nm])
            y_tr = np.array([1] * mask.sum() + [0] * n_nm)
            if len(np.unique(y_tr)) < 2:
                continue
            sc = StandardScaler()
            try:
                clf = _make_classifier()
                clf.fit(sc.fit_transform(F_tr), y_tr)
                probas[i] = clf.predict_proba(sc.transform(F_m[i : i + 1]))[0, 1]
            except Exception:
                pass
    else:
        from sklearn.model_selection import StratifiedKFold

        F_all = np.vstack([F_m, F_nm])
        y_all = np.array([1] * n_m + [0] * n_nm)
        for tr_idx, te_idx in StratifiedKFold(10, shuffle=True, random_state=42).split(F_all, y_all):
            mem_te = te_idx[te_idx < n_m]
            if len(mem_te) == 0:
                continue
            sc = StandardScaler()
            try:
                clf = _make_classifier()
                clf.fit(sc.fit_transform(F_all[tr_idx]), y_all[tr_idx])
                probas[mem_te] = clf.predict_proba(sc.transform(F_all[mem_te]))[:, 1]
            except Exception:
                pass

    # In-sample non-member scores for the FPR axis
    try:
        F_all   = np.vstack([F_m, F_nm])
        y_all   = np.array([1] * n_m + [0] * n_nm)
        sc      = StandardScaler()
        clf     = _make_classifier()
        clf.fit(sc.fit_transform(F_all), y_all)
        nm_probas   = clf.predict_proba(sc.transform(F_nm))[:, 1]
        full_probas = np.concatenate([probas, nm_probas])
        y_true      = np.array([1] * n_m + [0] * n_nm)
        return _tpr_at_fpr(y_true, full_probas, fpr_threshold=0.1)
    except Exception:
        return 0.0


# =============================================================================
#  Public API
# =============================================================================

def compute_mia(
    df_selected: pd.DataFrame,
    df_syn: pd.DataFrame,
    df_holdout: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    rng: np.random.Generator,
    n_null_runs: int = 200,
    max_non_members: Optional[int] = None,
    df_train: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run all three MIA layers and return their combined results.

    If df_train is not provided, Layers 1 and 2 are skipped and return
    AUC=0.5 (chance level) with p-value=1.0.

    Parameters
    ----------
    df_selected     : rows that were actually included in the prompt (members for Layer 0)
    df_syn          : synthetic data generated from the model
    df_holdout      : rows excluded from df_train (non-members for all layers)
    num_cols        : numeric column names
    cat_cols        : categorical column names
    rng             : random number generator
    n_null_runs     : number of permutation null runs per layer
    max_non_members : cap on non-member count; defaults to 3× members
    df_train        : full training set (members for Layers 1 and 2); optional

    Returns
    -------
    dict with keys: l0, l1, l2, privacy_score
    """
    if len(df_holdout) == 0:
        raise ValueError("df_holdout is empty.")

    # Align synthetic data columns to match real data
    df_syn_aligned = df_syn.copy()
    for col in num_cols + cat_cols:
        if col not in df_syn_aligned.columns:
            df_syn_aligned[col] = np.nan

    # --- Layer 0: Oracle -----------------------------------------------------
    l0 = compute_mia_layer0_oracle(
        df_selected=df_selected,
        df_syn_aligned=df_syn_aligned,
        df_holdout=df_holdout,
        num_cols=num_cols,
        cat_cols=cat_cols,
        rng=rng,
        n_null_runs=n_null_runs,
        max_non_members=max_non_members,
    )

    # --- Layers 1 & 2: only if df_train is available -------------------------
    if df_train is not None and len(df_train) > 0:
        l1 = compute_mia_layer1_knows_train(
            df_train=df_train,
            df_syn_aligned=df_syn_aligned,
            df_holdout=df_holdout,
            num_cols=num_cols,
            cat_cols=cat_cols,
            rng=rng,
            n_null_runs=n_null_runs,
            max_non_members=max_non_members,
        )
        l2 = compute_mia_layer2_blind(
            df_train=df_train,
            df_syn_aligned=df_syn_aligned,
            df_holdout=df_holdout,
            num_cols=num_cols,
            cat_cols=cat_cols,
            rng=rng,
            n_null_runs=n_null_runs,
            max_non_members=max_non_members,
        )
    else:
        # Layers 1 and 2 are unavailable without df_train
        l1 = dict(auc=0.5, null_mean=0.5, pvalue=1.0)
        l2 = dict(auc=0.5, null_mean=0.5, pvalue=1.0)

    # --- Privacy score -------------------------------------------------------
    # Measures how much the attack degrades from Layer 0 to Layer 2.
    # 1.0 = blind attack is no better than chance; 0.0 = no protection.
    oracle_lift   = l0["auc"] - 0.5
    blind_lift    = l2["auc"] - 0.5
    privacy_score = float(np.clip(1.0 - blind_lift / (oracle_lift + 1e-8), 0.0, 1.0))

    return dict(l0=l0, l1=l1, l2=l2, privacy_score=privacy_score)


def evaluate_attribution(
    df_selected: pd.DataFrame,
    df_syn: pd.DataFrame,
    df_holdout: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    strategy: str,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
    n_as_permutations: int = 1000,
    n_mia_null_runs: int = 200,
    max_non_members: Optional[int] = None,
    df_train: Optional[pd.DataFrame] = None,
    compute_attribution_score: bool = False,
) -> MIAMetrics:
    """
    Full evaluation: Attribution Score (optional) + all three MIA layers.

    Parameters
    ----------
    df_selected               : rows used in the prompt
    df_syn                    : synthetic data
    df_holdout                : held-out real rows (non-members)
    num_cols                  : numeric column names
    cat_cols                  : categorical column names
    strategy                  : experiment strategy label
    batch_size                : prompt batch size
    rng                       : random generator; defaults to seed 42
    n_as_permutations         : permutations for Attribution Score null distribution
    n_mia_null_runs           : permutations for MIA null distributions
    max_non_members           : cap on non-member count per MIA layer
    df_train                  : full training set for Layers 1 & 2; optional
    compute_attribution_score : if True, compute Attribution Score and related
                                metrics (intended for few-shot strategies only)

    Returns
    -------
    MIAMetrics
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(df_syn) == 0:
        raise ValueError("df_syn is empty.")

    df_syn = df_syn.copy()
    for col in num_cols + cat_cols:
        if col not in df_syn.columns:
            df_syn[col] = np.nan

    # --- Attribution Score (few-shot only, when explicitly enabled) ----------
    if compute_attribution_score:
        (AS, as_null_mean, as_null_std, as_pvalue,
         as_ci_low, as_ci_high,
         mean_d_own, mean_d_other,
         batches) = compute_batch_attribution_score(
            df_selected=df_selected,
            df_syn=df_syn,
            num_cols=num_cols,
            cat_cols=cat_cols,
            batch_size=batch_size,
            rng=rng,
            n_permutations=n_as_permutations,
        )
    else:
        AS           = float("nan")
        as_null_mean = float("nan")
        as_null_std  = float("nan")
        as_pvalue    = float("nan")
        as_ci_low    = float("nan")
        as_ci_high   = float("nan")
        mean_d_own   = float("nan")
        mean_d_other = float("nan")
        batches      = _batch_slices(len(df_selected), batch_size)

    # --- MIA (all three layers) ----------------------------------------------
    mia = compute_mia(
        df_selected=df_selected,
        df_syn=df_syn,
        df_holdout=df_holdout,
        num_cols=num_cols,
        cat_cols=cat_cols,
        rng=rng,
        n_null_runs=n_mia_null_runs,
        max_non_members=max_non_members,
        df_train=df_train,
    )
    l0 = mia["l0"]
    l1 = mia["l1"]
    l2 = mia["l2"]

    return MIAMetrics(
        strategy=strategy,
        n_examples=len(df_selected),
        batch_size=batch_size,
        n_batches=len(batches),
        # Attribution Score
        attribution_score=AS,
        attribution_score_null_mean=as_null_mean,
        attribution_score_null_std=as_null_std,
        attribution_score_pvalue=as_pvalue,
        attribution_score_ci_low=as_ci_low,
        attribution_score_ci_high=as_ci_high,
        mean_dist_to_own_batch=mean_d_own,
        mean_dist_to_other_batches=mean_d_other,
        # MIA Layer 0
        mia_auc=l0["auc"],
        mia_auc_null_mean=l0["null_mean"],
        mia_auc_null_std=l0["null_std"],
        mia_auc_pvalue=l0["pvalue"],
        mia_auc_ci_low=l0["ci_low"],
        mia_auc_ci_high=l0["ci_high"],
        mia_tpr_at_fpr10=l0["tpr_at_fpr10"],
        mia_matching_max_smd=l0["max_smd"],
        # MIA Layer 1
        mia_layer1_auc=l1["auc"],
        mia_layer1_null_mean=l1["null_mean"],
        mia_layer1_pvalue=l1["pvalue"],
        # MIA Layer 2
        mia_layer2_auc=l2["auc"],
        mia_layer2_null_mean=l2["null_mean"],
        mia_layer2_pvalue=l2["pvalue"],
        # Privacy
        mia_privacy_score=mia["privacy_score"],
        n_members=l0["n_members"],
        n_non_members=l0["n_non_members"],
    )
