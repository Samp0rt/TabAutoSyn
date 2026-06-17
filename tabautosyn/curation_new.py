"""
Synthetic Data Curation Pipeline.

Three-stage pipeline for curating synthetic datasets by iteratively removing
low-quality rows to maximise downstream model performance on real data.

Stages:
    1. VerifiedConfidenceFilter — drop lowest-confidence synthetic rows.
    2. BackwardElimination — block-level greedy removal with forward recovery.
    3. TargetedRowRefinement — fine-grained single-row and pair removal.

Scoring is pluggable: any ``(X_train, y_train, X_test, y_test) -> float``
callable works.  Defaults are provided for classification (LR + XGBoost
ROC AUC sum) and regression (LinearRegression + XGBoost R² sum).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
RICH_CONSOLE = Console()

ScoringFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]

_FAIL_SCORE = -1e6
_MIN_CLASSES = 2
_MIN_CLASS_SAMPLES = 3
_MIN_ACTIVE_ROWS = 30


def _curation_log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        RICH_CONSOLE.print(message)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute ROC AUC, returning ``None`` on degenerate inputs."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < _MIN_CLASSES:
        return None
    try:
        if y_score.ndim == 1:
            return float(roc_auc_score(y_true, y_score))
        if y_score.shape[1] == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _suppress_warnings():
    """Context-free helper — call inside ``warnings.catch_warnings()``."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _fit_lr(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(solver="lbfgs", max_iter=100, random_state=42)
    lr.fit(X_train, y_train)
    return lr


def _fit_xgb_clf(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=20, max_depth=3, learning_rate=0.2,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf


def _fit_xgb_reg(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> xgb.XGBRegressor:
    defaults = dict(n_estimators=20, max_depth=3, learning_rate=0.2, verbosity=0, random_state=42)
    defaults.update(kwargs)
    reg = xgb.XGBRegressor(**defaults)
    reg.fit(X_train, y_train)
    return reg


# ---------------------------------------------------------------------------
# Classification scorers
# ---------------------------------------------------------------------------

def score_lr_xgb_auc_sum(X_train, y_train, X_test, y_test) -> float:
    """Sum of Logistic Regression and XGBoost ROC AUC."""
    if len(np.unique(y_train)) < _MIN_CLASSES:
        return _FAIL_SCORE
    with warnings.catch_warnings():
        _suppress_warnings()
        try:
            proba_lr = _fit_lr(X_train, y_train).predict_proba(X_test)
        except Exception:
            return _FAIL_SCORE
        proba_xgb = _fit_xgb_clf(X_train, y_train).predict_proba(X_test)
    auc_lr = _safe_auc(y_test, proba_lr)
    auc_xgb = _safe_auc(y_test, proba_xgb)
    if auc_lr is None or auc_xgb is None:
        return _FAIL_SCORE
    return float(auc_lr + auc_xgb)


def score_xgb_auc(X_train, y_train, X_test, y_test) -> float:
    """Single XGBoost ROC AUC."""
    if len(np.unique(y_train)) < _MIN_CLASSES:
        return _FAIL_SCORE
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        proba = _fit_xgb_clf(X_train, y_train).predict_proba(X_test)
    auc = _safe_auc(y_test, proba)
    return float(auc) if auc is not None else _FAIL_SCORE


def score_lr_auc(X_train, y_train, X_test, y_test) -> float:
    """Single Logistic Regression ROC AUC."""
    if len(np.unique(y_train)) < _MIN_CLASSES:
        return _FAIL_SCORE
    with warnings.catch_warnings():
        _suppress_warnings()
        try:
            proba = _fit_lr(X_train, y_train).predict_proba(X_test)
        except Exception:
            return _FAIL_SCORE
    auc = _safe_auc(y_test, proba)
    return float(auc) if auc is not None else _FAIL_SCORE


# ---------------------------------------------------------------------------
# Regression scorers
# ---------------------------------------------------------------------------

def score_lr_xgb_r2_sum(X_train, y_train, X_test, y_test) -> float:
    """Sum of LinearRegression and XGBoost R²."""
    with warnings.catch_warnings():
        _suppress_warnings()
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            r2_lr = r2_score(y_test, lr.predict(X_test))
        except Exception:
            return _FAIL_SCORE
        r2_xgb = r2_score(y_test, _fit_xgb_reg(X_train, y_train).predict(X_test))
    return float(r2_lr + r2_xgb)


def score_xgb_r2(X_train, y_train, X_test, y_test) -> float:
    """Single XGBoost R²."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return float(r2_score(y_test, _fit_xgb_reg(X_train, y_train).predict(X_test)))


def score_lr_r2(X_train, y_train, X_test, y_test) -> float:
    """Single LinearRegression R²."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            return float(r2_score(y_test, lr.predict(X_test)))
        except Exception:
            return _FAIL_SCORE


def _target_counts_text(y: np.ndarray) -> str:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return "{" + ", ".join(f"{v}: {c}" for v, c in zip(values, counts)) + "}"


def _finite_summary_text(name: str, arr: np.ndarray) -> str:
    try:
        finite_mask = np.isfinite(arr)
        n_total = int(arr.size)
        n_finite = int(finite_mask.sum())
        return f"{name}: shape={arr.shape}, finite={n_finite}/{n_total}"
    except Exception as exc:
        return f"{name}: shape={getattr(arr, 'shape', '?')}, finite_check_failed={type(exc).__name__}: {exc}"


def _diagnose_fail_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> list[str]:
    """Explain why the default curation scorer returned ``_FAIL_SCORE``."""
    lines = [
        "[Curation][ScoreDebug] Baseline scorer returned fail score.",
        f"[Curation][ScoreDebug] task={task}",
        f"[Curation][ScoreDebug] {_finite_summary_text('X_train', X_train)}",
        f"[Curation][ScoreDebug] {_finite_summary_text('X_test', X_test)}",
        f"[Curation][ScoreDebug] y_train classes={_target_counts_text(y_train)}",
        f"[Curation][ScoreDebug] y_test classes={_target_counts_text(y_test)}",
    ]

    if task == "classification":
        if len(np.unique(y_train)) < _MIN_CLASSES:
            lines.append("[Curation][ScoreDebug] reason=single_class_y_train")
            return lines
        if len(np.unique(y_test)) < _MIN_CLASSES:
            lines.append("[Curation][ScoreDebug] reason=single_class_y_test_auc_undefined")
            return lines

        with warnings.catch_warnings():
            _suppress_warnings()
            try:
                lr = _fit_lr(X_train, y_train)
                proba_lr = lr.predict_proba(X_test)
                auc_lr = _safe_auc(y_test, proba_lr)
                lines.append(
                    f"[Curation][ScoreDebug] logistic_regression=ok, auc={auc_lr}"
                )
            except Exception as exc:
                lines.append(
                    "[Curation][ScoreDebug] reason=logistic_regression_failed "
                    f"{type(exc).__name__}: {exc}"
                )
                return lines

            try:
                xgb_clf = _fit_xgb_clf(X_train, y_train)
                proba_xgb = xgb_clf.predict_proba(X_test)
                auc_xgb = _safe_auc(y_test, proba_xgb)
                lines.append(f"[Curation][ScoreDebug] xgboost=ok, auc={auc_xgb}")
            except Exception as exc:
                lines.append(
                    "[Curation][ScoreDebug] reason=xgboost_failed "
                    f"{type(exc).__name__}: {exc}"
                )
                return lines

        if auc_lr is None or auc_xgb is None:
            lines.append("[Curation][ScoreDebug] reason=auc_failed_or_undefined")
        return lines

    with warnings.catch_warnings():
        _suppress_warnings()
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            r2_lr = r2_score(y_test, lr.predict(X_test))
            lines.append(f"[Curation][ScoreDebug] linear_regression=ok, r2={r2_lr}")
        except Exception as exc:
            lines.append(
                "[Curation][ScoreDebug] reason=linear_regression_failed "
                f"{type(exc).__name__}: {exc}"
            )
            return lines

        try:
            pred_xgb = _fit_xgb_reg(X_train, y_train).predict(X_test)
            r2_xgb = r2_score(y_test, pred_xgb)
            lines.append(f"[Curation][ScoreDebug] xgboost_regression=ok, r2={r2_xgb}")
        except Exception as exc:
            lines.append(
                "[Curation][ScoreDebug] reason=xgboost_regression_failed "
                f"{type(exc).__name__}: {exc}"
            )

    return lines


# ---------------------------------------------------------------------------
# Fast (lightweight) scorer factory
# ---------------------------------------------------------------------------

def _make_fast_scorer(scoring_fn: ScoringFn, task: str = "classification") -> ScoringFn:
    """Build a cheap XGBoost-only scorer for candidate prescoring."""
    if task == "regression":
        def _fast(X_train, y_train, X_test, y_test):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                pred = _fit_xgb_reg(X_train, y_train, n_estimators=10, max_depth=2, learning_rate=0.3).predict(X_test)
            return float(r2_score(y_test, pred))
        return _fast

    def _fast(X_train, y_train, X_test, y_test):
        if len(np.unique(y_train)) < _MIN_CLASSES:
            return _FAIL_SCORE
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf = xgb.XGBClassifier(
                n_estimators=10, max_depth=2, learning_rate=0.3,
                use_label_encoder=False, eval_metric="logloss", verbosity=0,
            )
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)
        auc = _safe_auc(y_test, proba)
        return float(auc) if auc is not None else _FAIL_SCORE
    return _fast


# ---------------------------------------------------------------------------
# Data preparation utilities
# ---------------------------------------------------------------------------

def _prepare_arrays(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    task: str = "classification",
):
    X_syn = syn_df[feature_cols].astype(float).values
    X_real = real_df[feature_cols].astype(float).values

    if task == "regression":
        y_syn = syn_df[target_col].astype(float).values
        y_real = real_df[target_col].astype(float).values
        return X_syn, y_syn, X_real, y_real, None

    le = LabelEncoder()
    le.fit(np.concatenate([syn_df[target_col].values, real_df[target_col].values]))
    y_syn = le.transform(syn_df[target_col].values)
    y_real = le.transform(real_df[target_col].values)
    return X_syn, y_syn, X_real, y_real, le


def _has_all_classes(y: np.ndarray, idx: np.ndarray, n_classes: int) -> bool:
    return len(np.unique(y[idx])) >= n_classes


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def _get_confidence_scores(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    task: str = "classification",
) -> np.ndarray:
    """Per-row confidence: P(correct class) for classification, 1/(1+|residual|) for regression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if task == "regression":
            pred = _fit_xgb_reg(
                X_real, y_real, n_estimators=50, max_depth=4, learning_rate=0.1,
            ).predict(X_syn)
            return 1.0 / (1.0 + np.abs(y_syn - pred))

        clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        clf.fit(X_real, y_real)
        proba = clf.predict_proba(X_syn)
        return proba[np.arange(len(y_syn)), y_syn]


def _min_retained_rows(initial_size: int, min_rows_ratio: float) -> int:
    """Minimum row count to keep based on the original pipeline input size."""
    if initial_size <= 0:
        return _MIN_ACTIVE_ROWS
    return max(_MIN_ACTIVE_ROWS, int(initial_size * min_rows_ratio))


def _apply_mask(active_mask: np.ndarray, removal: frozenset) -> np.ndarray:
    """Return a copy of *active_mask* with *removal* indices set to False."""
    trial = active_mask.copy()
    for idx in removal:
        trial[idx] = False
    return trial


# ---------------------------------------------------------------------------
# Stage 1: Verified Confidence Filter
# ---------------------------------------------------------------------------

class VerifiedConfidenceFilter:
    """Drop the lowest-confidence quantile of synthetic rows (with rollback)."""

    def __init__(self, drop_quantile: float = 0.10):
        self.drop_quantile = drop_quantile

    def curate(
        self,
        syn_data: pd.DataFrame,
        real_data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        scoring_fn: ScoringFn,
        task: str = "classification",
        min_retained_rows: int = _MIN_ACTIVE_ROWS,
        verbose: bool = True,
    ) -> pd.DataFrame:
        syn_data = _drop_duplicates(syn_data)
        if len(syn_data) <= min_retained_rows:
            return syn_data

        X_syn, y_syn, X_real, y_real, _ = _prepare_arrays(
            syn_data, real_data, feature_cols, target_col, task,
        )
        score_before = scoring_fn(X_syn, y_syn, X_real, y_real)

        confidence = _get_confidence_scores(X_syn, y_syn, X_real, y_real, task)
        max_removable = len(syn_data) - min_retained_rows
        if max_removable <= 0:
            return syn_data

        effective_drop = min(
            self.drop_quantile,
            max_removable / len(syn_data),
        )
        cutoff = np.quantile(confidence, effective_drop)
        mask = confidence >= cutoff

        if task == "classification":
            self._ensure_min_class_samples(y_syn, confidence, mask)

        if mask.sum() < min_retained_rows:
            keep_idx = np.argsort(confidence)[-min_retained_rows:]
            mask = np.zeros(len(syn_data), dtype=bool)
            mask[keep_idx] = True
            if task == "classification":
                self._ensure_min_class_samples(y_syn, confidence, mask)

        candidate = syn_data.iloc[mask].reset_index(drop=True)
        X_c, y_c, _, _, _ = _prepare_arrays(candidate, real_data, feature_cols, target_col, task)
        score_after = scoring_fn(X_c, y_c, X_real, y_real)

        if score_after >= score_before:
            _curation_log(
                f"[dim]  Confidence filter[/dim]  "
                f"{len(syn_data):,} → {len(candidate):,} rows  "
                f"score {score_before:.4f} → [green]{score_after:.4f}[/green]  "
                f"[dim](applied)[/dim]",
                verbose=verbose,
            )
            return candidate

        _curation_log(
            f"[dim]  Confidence filter[/dim]  "
            f"score {score_before:.4f} → {score_after:.4f}  "
            f"[yellow]rolled back[/yellow]",
            verbose=verbose,
        )
        return syn_data

    @staticmethod
    def _ensure_min_class_samples(
        y: np.ndarray, confidence: np.ndarray, mask: np.ndarray,
    ) -> None:
        """Keep at least ``_MIN_CLASS_SAMPLES`` rows per class."""
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if mask[cls_idx].sum() < _MIN_CLASS_SAMPLES:
                top_k = cls_idx[np.argsort(confidence[cls_idx])[-_MIN_CLASS_SAMPLES:]]
                mask[top_k] = True


# ---------------------------------------------------------------------------
# Stage 2: Backward Elimination
# ---------------------------------------------------------------------------

class BackwardElimination:
    """Block-level greedy removal with forward recovery.

    Adaptive strategy:
        * < 1000 rows — full scoring for every candidate.
        * >= 1000 rows — fast prescore, then verify top-K with full scorer.

    Uses both random and confidence-guided partitions.
    """

    def __init__(
        self,
        n_partitions_random: int = 3,
        n_partitions_guided: int = 2,
        top_k_verify: int = 5,
        patience: int = 2,
        min_retained_rows: int = _MIN_ACTIVE_ROWS,
    ):
        self.n_partitions_random = n_partitions_random
        self.n_partitions_guided = n_partitions_guided
        self.top_k_verify = top_k_verify
        self.patience = patience
        self.min_retained_rows = min_retained_rows

    # -- helpers --

    @staticmethod
    def _adaptive_block_sizes(n_rows: int) -> List[int]:
        if n_rows < 300:
            return [10, 15, 20]
        if n_rows < 1000:
            return [15, 20, 30]
        if n_rows < 3000:
            return [20, 30, 50]
        return [30, 50, 80]

    @staticmethod
    def _assign_blocks_random(
        active_idx: np.ndarray,
        y: np.ndarray,
        n_blocks: int,
        rng: np.random.RandomState,
        task: str,
    ) -> np.ndarray:
        block_ids = np.full(len(y), -1, dtype=int)
        if task == "classification":
            for cls in np.unique(y[active_idx]):
                cls_active = [i for i in active_idx if y[i] == cls]
                rng.shuffle(cls_active)
                for i, idx in enumerate(cls_active):
                    block_ids[idx] = i % n_blocks
        else:
            shuffled = list(active_idx)
            rng.shuffle(shuffled)
            for i, idx in enumerate(shuffled):
                block_ids[idx] = i % n_blocks
        return block_ids

    @staticmethod
    def _assign_blocks_guided(
        active_idx: np.ndarray,
        y: np.ndarray,
        n_blocks: int,
        confidence: np.ndarray,
        rng: np.random.RandomState,
        task: str,
    ) -> np.ndarray:
        block_ids = np.full(len(y), -1, dtype=int)
        jitter_prob = 0.15

        def _fill(indices: np.ndarray) -> None:
            confs = confidence[indices]
            sorted_idx = indices[np.argsort(confs)]
            chunk = max(1, len(sorted_idx) // n_blocks)
            for i, idx in enumerate(sorted_idx):
                block = min(i // chunk, n_blocks - 1)
                if rng.random() < jitter_prob and block > 0:
                    block -= 1
                block_ids[idx] = block

        if task == "classification":
            for cls in np.unique(y[active_idx]):
                _fill(np.array([i for i in active_idx if y[i] == cls]))
        else:
            _fill(active_idx)

        return block_ids

    def _generate_removals(
        self,
        active_mask: np.ndarray,
        y: np.ndarray,
        block_sizes: List[int],
        confidence: np.ndarray,
        round_num: int,
        task: str,
    ) -> List[frozenset]:
        """Generate deduplicated block-removal candidates across all partitions."""
        n_classes = len(np.unique(y[active_mask])) if task == "classification" else 0
        active_idx = np.where(active_mask)[0]
        min_rows = self.min_retained_rows

        seen: set[frozenset] = set()
        removals: list[frozenset] = []

        for n_blocks in block_sizes:
            actual = min(n_blocks, len(active_idx) // 3)
            if actual < 2:
                continue

            partitions = []
            for ps in range(self.n_partitions_random):
                rng = np.random.RandomState(42 + round_num * 1000 + ps * 17 + n_blocks)
                partitions.append(
                    self._assign_blocks_random(active_idx, y, actual, rng, task)
                )
            for ps in range(self.n_partitions_guided):
                rng = np.random.RandomState(99 + round_num * 1000 + ps * 31 + n_blocks)
                partitions.append(
                    self._assign_blocks_guided(active_idx, y, actual, confidence, rng, task)
                )

            for block_ids in partitions:
                for block in np.unique(block_ids[active_mask & (block_ids >= 0)]):
                    removal = frozenset(np.where(active_mask & (block_ids == block))[0])
                    if removal in seen:
                        continue
                    seen.add(removal)

                    if len(active_idx) - len(removal) < min_rows:
                        continue
                    if task == "classification":
                        trial = _apply_mask(active_mask, removal)
                        if not _has_all_classes(y, np.where(trial)[0], n_classes):
                            continue
                    removals.append(removal)

        return removals

    # -- main loop --

    def curate(
        self,
        syn_data: pd.DataFrame,
        real_data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        scoring_fn: ScoringFn,
        fast_scoring_fn: ScoringFn,
        task: str = "classification",
        min_retained_rows: int = _MIN_ACTIVE_ROWS,
        verbose: bool = True,
    ) -> pd.DataFrame:
        syn_data = _drop_duplicates(syn_data)
        X, y, Xr, yr, _ = _prepare_arrays(syn_data, real_data, feature_cols, target_col, task)

        n = len(X)
        n_classes = len(np.unique(y)) if task == "classification" else 0
        block_sizes = self._adaptive_block_sizes(n)
        use_fast = n >= 1000
        min_rows = min(min_retained_rows, n)

        active_mask = np.ones(n, dtype=bool)
        current_score = scoring_fn(X, y, Xr, yr)
        current_fast = fast_scoring_fn(X, y, Xr, yr) if use_fast else 0.0

        removed_history: list[frozenset] = []
        no_improve = 0
        round_num = 0

        while no_improve < self.patience:
            round_num += 1
            active_idx = np.where(active_mask)[0]
            if len(active_idx) <= min_rows:
                break

            confidence = np.full(n, 1.0)
            confidence[active_idx] = _get_confidence_scores(
                X[active_idx], y[active_idx], Xr, yr, task,
            )

            removals = self._generate_removals(active_mask, y, block_sizes, confidence, round_num, task)
            if not removals:
                no_improve += 1
                continue

            to_verify = self._prescore(removals, active_mask, X, y, Xr, yr, fast_scoring_fn, current_fast) if use_fast else removals
            best_removal, best_score = self._pick_best(to_verify, active_mask, X, y, Xr, yr, scoring_fn, current_score)

            if best_removal is not None:
                active_mask = _apply_mask(active_mask, best_removal)
                delta = best_score - current_score
                current_score = best_score
                removed_history.append(best_removal)
                if use_fast:
                    idx = np.where(active_mask)[0]
                    current_fast = fast_scoring_fn(X[idx], y[idx], Xr, yr)
                _curation_log(
                    f"[dim]  Block removal[/dim]  round {round_num}  "
                    f"removed {len(best_removal):,} rows  "
                    f"score [green]{current_score:.4f}[/green] "
                    f"[dim](+{delta:.4f})[/dim]",
                    verbose=verbose,
                )
                no_improve = 0
            else:
                no_improve += 1

        current_score = self._forward_recovery(
            removed_history, active_mask, X, y, Xr, yr, n_classes, scoring_fn, current_score, task, verbose=verbose,
        )

        return syn_data.iloc[np.where(active_mask)[0]].reset_index(drop=True)

    def _prescore(
        self, removals, active_mask, X, y, Xr, yr, fast_fn, current_fast,
    ) -> list[frozenset]:
        scored = []
        for removal in removals:
            trial_idx = np.where(_apply_mask(active_mask, removal))[0]
            fs = fast_fn(X[trial_idx], y[trial_idx], Xr, yr)
            scored.append((fs - current_fast, removal))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:self.top_k_verify]]

    @staticmethod
    def _pick_best(candidates, active_mask, X, y, Xr, yr, scoring_fn, current_score):
        best_removal = None
        best_score = current_score
        for removal in candidates:
            trial_idx = np.where(_apply_mask(active_mask, removal))[0]
            score = scoring_fn(X[trial_idx], y[trial_idx], Xr, yr)
            if score > best_score:
                best_score = score
                best_removal = removal
        return best_removal, best_score

    def _forward_recovery(
        self, history, active_mask, X, y, Xr, yr, n_classes, scoring_fn, current_score, task, *, verbose: bool = True,
    ) -> float:
        if len(history) <= 1:
            return current_score
        for i, block in enumerate(history):
            trial = active_mask.copy()
            for idx in block:
                trial[idx] = True
            trial_idx = np.where(trial)[0]
            if task == "classification" and not _has_all_classes(y, trial_idx, n_classes):
                continue
            score = scoring_fn(X[trial_idx], y[trial_idx], Xr, yr)
            if score > current_score:
                active_mask[:] = trial
                current_score = score
                _curation_log(
                    f"[dim]  Block removal[/dim]  restored block {i} "
                    f"({len(block):,} rows)  score [green]{current_score:.4f}[/green]",
                    verbose=verbose,
                )
        return current_score


# ---------------------------------------------------------------------------
# Stage 3: Targeted Row Refinement
# ---------------------------------------------------------------------------

class TargetedRowRefinement:
    """Fine-grained removal: individual rows, then pairs from the low-confidence pool."""

    def __init__(self, max_passes: int = 3, min_retained_rows: int = _MIN_ACTIVE_ROWS):
        self.max_passes = max_passes
        self.min_retained_rows = min_retained_rows

    @staticmethod
    def _adaptive_params(n_rows: int) -> tuple[int, int, int]:
        singles = min(120, max(40, n_rows // 5))
        groups = min(50, max(15, n_rows // 10))
        patience = min(25, max(12, n_rows // 20))
        return singles, groups, patience

    def curate(
        self,
        syn_data: pd.DataFrame,
        real_data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        scoring_fn: ScoringFn,
        fast_scoring_fn: ScoringFn,
        task: str = "classification",
        verbose: bool = True,
    ) -> pd.DataFrame:
        syn_data = _drop_duplicates(syn_data)
        X, y, Xr, yr, _ = _prepare_arrays(syn_data, real_data, feature_cols, target_col, task)

        n = len(X)
        n_classes = len(np.unique(y)) if task == "classification" else 0
        max_singles, max_groups, patience = self._adaptive_params(n)
        use_fast = n >= 1000
        min_rows = min(self.min_retained_rows, n)

        active_mask = np.ones(n, dtype=bool)
        current_score = scoring_fn(X[active_mask], y[active_mask], Xr, yr)
        total_removed = 0

        for pass_num in range(self.max_passes):
            active_idx = np.where(active_mask)[0]
            if len(active_idx) <= min_rows:
                break

            confidence = np.full(n, 1.0)
            confidence[active_idx] = _get_confidence_scores(X[active_idx], y[active_idx], Xr, yr, task)
            candidates = active_idx[np.argsort(confidence[active_idx])]

            if use_fast:
                pass_removed, current_score = self._singles_fast(
                    candidates, active_mask, X, y, Xr, yr,
                    scoring_fn, fast_scoring_fn, max_singles, patience, task, min_rows,
                )
            else:
                pass_removed, current_score = self._singles_full(
                    candidates, active_mask, X, y, Xr, yr,
                    scoring_fn, max_singles, patience, task, min_rows,
                )

            pass_removed += self._pairs(
                active_mask, X, y, Xr, yr, scoring_fn, n_classes,
                max_groups, patience, pass_num, task, min_rows,
            )[0]
            current_score = scoring_fn(X[active_mask], y[active_mask], Xr, yr)

            total_removed += pass_removed
            if pass_removed == 0:
                break
            _curation_log(
                f"[dim]  Row refinement[/dim]  pass {pass_num + 1}  "
                f"removed {pass_removed:,} rows  "
                f"score [green]{current_score:.4f}[/green]",
                verbose=verbose,
            )

        if total_removed > 0:
            _curation_log(
                f"[dim]  Row refinement[/dim]  total removed: {total_removed:,} rows",
                verbose=verbose,
            )

        return syn_data.iloc[np.where(active_mask)[0]].reset_index(drop=True)

    # -- single-row removal --

    def _singles_fast(
        self, candidates, active_mask, X, y, Xr, yr, scoring_fn, fast_fn,
        max_singles, patience, task, min_rows,
    ):
        current_fast = fast_fn(X[active_mask], y[active_mask], Xr, yr)
        current_score = scoring_fn(X[active_mask], y[active_mask], Xr, yr)
        scan_limit = min(max_singles * 2, len(candidates))
        removed = 0

        fast_results = []
        for row in candidates[:scan_limit]:
            if active_mask.sum() <= min_rows:
                break
            if task == "classification" and np.sum(y[active_mask] == y[row]) <= _MIN_CLASS_SAMPLES:
                continue
            trial = active_mask.copy()
            trial[row] = False
            fs = fast_fn(X[trial], y[trial], Xr, yr)
            fast_results.append((row, fs - current_fast))
        fast_results.sort(key=lambda x: x[1], reverse=True)

        no_improve = 0
        for row, fd in fast_results[:max_singles]:
            if no_improve >= patience or fd <= -0.01:
                break
            if active_mask.sum() <= min_rows:
                break
            if not active_mask[row]:
                continue
            trial = active_mask.copy()
            trial[row] = False
            score = scoring_fn(X[trial], y[trial], Xr, yr)
            if score > current_score:
                active_mask[:] = trial
                current_score = score
                current_fast = fast_fn(X[active_mask], y[active_mask], Xr, yr)
                removed += 1
                no_improve = 0
            else:
                no_improve += 1

        return removed, current_score

    def _singles_full(
        self, candidates, active_mask, X, y, Xr, yr, scoring_fn,
        max_singles, patience, task, min_rows,
    ):
        current_score = scoring_fn(X[active_mask], y[active_mask], Xr, yr)
        no_improve = 0
        tried = 0
        removed = 0

        for row in candidates:
            if tried >= max_singles or no_improve >= patience:
                break
            if active_mask.sum() <= min_rows:
                break
            if not active_mask[row]:
                continue
            if task == "classification" and np.sum(y[active_mask] == y[row]) <= _MIN_CLASS_SAMPLES:
                continue
            trial = active_mask.copy()
            trial[row] = False
            score = scoring_fn(X[trial], y[trial], Xr, yr)
            tried += 1
            if score > current_score:
                active_mask[:] = trial
                current_score = score
                removed += 1
                no_improve = 0
            else:
                no_improve += 1

        return removed, current_score

    # -- pair removal --

    def _pairs(
        self, active_mask, X, y, Xr, yr, scoring_fn, n_classes,
        max_groups, patience, pass_num, task, min_rows,
    ):
        active_idx = np.where(active_mask)[0]
        if len(active_idx) < 20 or len(active_idx) <= min_rows + 1:
            return 0, scoring_fn(X[active_mask], y[active_mask], Xr, yr)

        n = len(y)
        confidence = np.full(n, 1.0)
        confidence[active_idx] = _get_confidence_scores(X[active_idx], y[active_idx], Xr, yr, task)
        sorted_active = active_idx[np.argsort(confidence[active_idx])]
        pool = sorted_active[:max(3, len(sorted_active) // 5)]

        current_score = scoring_fn(X[active_mask], y[active_mask], Xr, yr)
        rng = np.random.RandomState(42 + pass_num)
        no_improve = 0
        tried = 0
        removed = 0

        for _ in range(max_groups * 3):
            if tried >= max_groups or no_improve >= patience:
                break
            if active_mask.sum() <= min_rows + 1:
                break
            pair = rng.choice(pool, size=2, replace=False)
            if not all(active_mask[g] for g in pair):
                continue
            if task == "classification":
                counts: dict[int, int] = {}
                for g in pair:
                    c = y[g]
                    counts[c] = counts.get(c, 0) + 1
                if not all(np.sum(y[active_mask] == c) - cnt >= _MIN_CLASS_SAMPLES for c, cnt in counts.items()):
                    continue
            trial = active_mask.copy()
            for g in pair:
                trial[g] = False
            trial_idx = np.where(trial)[0]
            if task == "classification" and not _has_all_classes(y, trial_idx, n_classes):
                continue
            score = scoring_fn(X[trial_idx], y[trial_idx], Xr, yr)
            tried += 1
            if score > current_score:
                active_mask[:] = trial
                current_score = score
                removed += 2
                no_improve = 0
            else:
                no_improve += 1

        return removed, current_score


# ---------------------------------------------------------------------------
# Pipeline config & orchestrator
# ---------------------------------------------------------------------------

@dataclass
class CuratorConfig:
    """Configuration for :class:`SyntheticDataCurator`.

    Args:
        task: ``"classification"`` or ``"regression"``.
        scoring_fn: Custom scorer ``(X_train, y_train, X_test, y_test) -> float``.
            Auto-selected by *task* when ``None``.
        fast_scoring_fn: Lightweight scorer for prescoring (auto-created when ``None``).
        confidence_drop: Quantile fraction to pre-filter via confidence.
        n_partitions_random: Random partition count per block size.
        n_partitions_guided: Confidence-guided partition count per block size.
        top_k_verify: Candidates to fully verify after fast prescoring.
        backward_patience: Rounds without improvement before stopping backward elimination.
        min_rows_ratio: Global floor — never drop below this fraction of the
            original input row count across all curation stages.
        refine_passes: Maximum passes of row-level refinement.
        verbose: Print progress to stdout.
    """

    task: str = "classification"
    scoring_fn: Optional[ScoringFn] = None
    fast_scoring_fn: Optional[ScoringFn] = None
    confidence_drop: float = 0.05
    n_partitions_random: int = 3
    n_partitions_guided: int = 2
    top_k_verify: int = 5
    backward_patience: int = 2
    min_rows_ratio: float = 0.75
    refine_passes: int = 3
    verbose: bool = True


class SyntheticDataCurator:
    """Three-stage curation pipeline with pluggable scoring.

    Examples::

        # Classification (default LR + XGBoost AUC sum)
        curator = SyntheticDataCurator()
        curated = curator.curate(syn_df, real_df, feature_cols, "target")

        # Regression (default LR + XGBoost R² sum)
        config = CuratorConfig(task="regression")
        curator = SyntheticDataCurator(config)
        curated = curator.curate(syn_df, real_df, feature_cols, "target")

        # Custom scorer
        def my_scorer(X_tr, y_tr, X_te, y_te):
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_tr, y_tr)
            return f1_score(y_te, clf.predict(X_te), average="macro")

        config = CuratorConfig(scoring_fn=my_scorer)
        curator = SyntheticDataCurator(config)
        curated = curator.curate(syn_df, real_df, feature_cols, "target")
    """

    _DEFAULT_SCORERS = {
        "classification": score_lr_xgb_auc_sum,
        "regression": score_lr_xgb_r2_sum,
    }

    def __init__(self, config: Optional[CuratorConfig] = None):
        self.config = config or CuratorConfig()
        self.task = self.config.task
        self.scoring_fn = self.config.scoring_fn or self._DEFAULT_SCORERS[self.task]
        self.fast_scoring_fn = self.config.fast_scoring_fn or _make_fast_scorer(self.scoring_fn, self.task)

    def curate(
        self,
        syn_data: pd.DataFrame,
        real_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "target",
    ) -> pd.DataFrame:
        if feature_cols is None:
            feature_cols = [c for c in syn_data.columns if c != target_col]

        current = _drop_duplicates(syn_data)
        initial_size = len(current)
        min_retained_rows = _min_retained_rows(initial_size, self.config.min_rows_ratio)

        X, y, Xr, yr, _ = _prepare_arrays(current, real_data, feature_cols, target_col, self.task)
        baseline = self.scoring_fn(X, y, Xr, yr)
        self._log(
            f"[cyan]Curation baseline[/cyan]: {initial_size:,} rows, "
            f"score={baseline:.4f}, min retained={min_retained_rows:,} "
            f"[dim]({100 * self.config.min_rows_ratio:.0f}% floor)[/dim]"
        )
        if baseline <= _FAIL_SCORE and self.config.verbose:
            for line in _diagnose_fail_score(X, y, Xr, yr, self.task):
                _curation_log(f"[yellow]{line}[/yellow]", verbose=True)

        current = self._run_confidence_filter(
            current, real_data, feature_cols, target_col, min_retained_rows,
        )
        current = self._run_backward_elimination(
            current, real_data, feature_cols, target_col, min_retained_rows,
        )
        current = self._run_row_refinement(
            current, real_data, feature_cols, target_col, min_retained_rows,
        )

        current = _drop_duplicates(current)
        self._log_final(current, real_data, feature_cols, target_col, initial_size, baseline)
        return current

    # -- stage runners --

    def _run_confidence_filter(
        self, current, real_data, feature_cols, target_col, min_retained_rows,
    ):
        if self.config.confidence_drop <= 0:
            return current
        cf = VerifiedConfidenceFilter(drop_quantile=self.config.confidence_drop)
        current = cf.curate(
            current,
            real_data,
            feature_cols,
            target_col,
            self.scoring_fn,
            self.task,
            min_retained_rows=min_retained_rows,
            verbose=self.config.verbose,
        )
        self._log_stage("confidence filter", current, real_data, feature_cols, target_col)
        return current

    def _run_backward_elimination(
        self, current, real_data, feature_cols, target_col, min_retained_rows,
    ):
        be = BackwardElimination(
            n_partitions_random=self.config.n_partitions_random,
            n_partitions_guided=self.config.n_partitions_guided,
            top_k_verify=self.config.top_k_verify,
            patience=self.config.backward_patience,
            min_retained_rows=min_retained_rows,
        )
        current = be.curate(
            current,
            real_data,
            feature_cols,
            target_col,
            self.scoring_fn,
            self.fast_scoring_fn,
            self.task,
            min_retained_rows=min_retained_rows,
            verbose=self.config.verbose,
        )
        self._log_stage("block removal", current, real_data, feature_cols, target_col)
        return current

    def _run_row_refinement(
        self, current, real_data, feature_cols, target_col, min_retained_rows,
    ):
        refiner = TargetedRowRefinement(
            max_passes=self.config.refine_passes,
            min_retained_rows=min_retained_rows,
        )
        current = refiner.curate(
            current,
            real_data,
            feature_cols,
            target_col,
            self.scoring_fn,
            self.fast_scoring_fn,
            self.task,
            verbose=self.config.verbose,
        )
        return current

    # -- logging helpers --

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            RICH_CONSOLE.print(msg)

    def _log_stage(self, name: str, current, real_data, feature_cols, target_col) -> None:
        if not self.config.verbose:
            return
        X, y, Xr, yr, _ = _prepare_arrays(current, real_data, feature_cols, target_col, self.task)
        s = self.scoring_fn(X, y, Xr, yr)
        RICH_CONSOLE.print(
            f"[cyan]After {name}[/cyan]: [bold]{len(current):,}[/bold] rows, "
            f"score={s:.4f}"
        )

    def _log_final(self, current, real_data, feature_cols, target_col, initial_size, baseline) -> None:
        if not self.config.verbose:
            return
        X, y, Xr, yr, _ = _prepare_arrays(current, real_data, feature_cols, target_col, self.task)
        final = self.scoring_fn(X, y, Xr, yr)
        RICH_CONSOLE.print(
            f"[green]Curation finished[/green]: "
            f"[bold]{len(current):,}[/bold]/{initial_size:,} rows retained "
            f"({100 * len(current) / initial_size:.1f}%), "
            f"score={final:.4f} [dim](Δ{final - baseline:+.4f})[/dim]"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def curate_synthetic_data(
    syn_data: pd.DataFrame,
    real_data: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target",
    task: str = "classification",
    scoring_fn: Optional[ScoringFn] = None,
    confidence_drop: float = 0.05,
    min_rows_ratio: float = 0.75,
    verbose: bool = True,
) -> pd.DataFrame:
    """One-liner convenience wrapper around :class:`SyntheticDataCurator`.

    Examples::

        # Classification (default)
        curated = curate_synthetic_data(syn_df, real_df, target_col="label")

        # Regression
        curated = curate_synthetic_data(syn_df, real_df, target_col="price", task="regression")
    """
    config = CuratorConfig(
        task=task,
        scoring_fn=scoring_fn,
        confidence_drop=confidence_drop,
        min_rows_ratio=min_rows_ratio,
        verbose=verbose,
    )
    return SyntheticDataCurator(config).curate(syn_data, real_data, feature_cols, target_col)
