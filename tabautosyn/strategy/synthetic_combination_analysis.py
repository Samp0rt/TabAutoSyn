"""
synthetic_combination_analysis.py
==================================
Decomposes synthetic data rows as weighted combinations of real data rows,
using sparsity-regularized convex optimization (or NNLS as a fallback).

Pipeline:
  1. check_overdetermination   – basis diagnostics (rank, condition number)
  2. fit_combination_regularized – per-row optimization: min ‖Xᵀα − x_syn‖² + λ·H(α)
  3. analyze_run                – applies step 2 across all synthetic rows in one run
  4. analyze_all_runs           – aggregates multiple LLM runs; normalizes via StandardScaler
  5. test_noise_structure       – per-feature residual analysis (bias t-test, normality)
  6. plot_analysis              – 6-panel diagnostic figure
  7. run_full_analysis          – end-to-end entry point
"""

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import OptimizeWarning, minimize
from scipy.stats import normaltest, shapiro, skew, ttest_1samp
from sklearn.preprocessing import StandardScaler

# Suppress optimizer convergence warnings only; let all other warnings through.
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum ‖x_syn‖ (in StandardScaler units) below which NRMSE is unreliable.
_NEAR_ZERO_THRESHOLD: float = 0.5

# Significance level for bias t-test and normality test.
_ALPHA_SIGNIFICANCE: float = 0.05

# Threshold for "dominant" weight: α_i > _DOMINANT_THRESHOLD counts as a
# meaningful contributor.
_DOMINANT_THRESHOLD: float = 0.1

# NRMSE and cosine similarity acceptance thresholds for the final verdict.
_NRMSE_THRESHOLD: float = 0.3
_COSINE_THRESHOLD: float = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basis diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def check_overdetermination(X_real: np.ndarray) -> dict:
    """
    Inspect whether X_real is a suitable basis for decomposition.

    If rank == n_real and n_real == n_features the system is square,
    so a trivial zero-residual solution exists (R² ≡ 1).

    Parameters
    ----------
    X_real : ndarray of shape (n_real, n_features)

    Returns
    -------
    dict with keys:
        n_real, n_features, rank, singular_values, condition_number,
        is_overdetermined, is_underdetermined, effective_dof
    """
    n_real, n_features = X_real.shape
    rank = np.linalg.matrix_rank(X_real)
    _, singular_values, _ = np.linalg.svd(X_real, full_matrices=False)
    condition_number = singular_values[0] / (singular_values[-1] + 1e-12)

    return {
        "n_real": n_real,
        "n_features": n_features,
        "rank": rank,
        "singular_values": singular_values,
        "condition_number": condition_number,
        # Overdetermined (features > samples): NNLS has a non-trivial solution.
        "is_overdetermined": n_features > n_real,
        # Underdetermined (features ≤ samples): trivial R²≈1 at λ=0.
        "is_underdetermined": n_features <= n_real,
        "effective_dof": n_features - n_real,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sparsity-regularized decomposition
# ─────────────────────────────────────────────────────────────────────────────

def fit_combination_regularized(
    x_syn_scaled: np.ndarray,
    X_real_scaled: np.ndarray,
    lam_sparsity: float = 0.05,
    convex: bool = True,
) -> dict:
    """
    Find weights α that best approximate x_syn as a combination of real rows.

    Objective
    ---------
        min_α  ‖X_real^T α − x_syn‖²  +  λ · H(α)

    where H(α) = −Σ αᵢ log αᵢ is the entropy of α.

    Minimizing entropy penalizes uniform distributions and pushes α toward
    sparse solutions—this differs from MaxEnt / RL-style regularization which
    *maximizes* entropy.  Sparse α reveals which real examples the LLM
    actually relies on when generating a synthetic row.

    Parameters
    ----------
    x_syn_scaled  : 1-D array, shape (n_features,)
    X_real_scaled : 2-D array, shape (n_real, n_features)
    lam_sparsity  : sparsity penalty λ.  0 → pure NNLS / convex QP.
                    Recommended range: 0.01–0.1.
    convex        : if True, enforce α ≥ 0, Σα = 1 (SLSQP);
                    if False, enforce α ≥ 0 only (L-BFGS-B).

    Returns
    -------
    dict with reconstruction metrics and the optimized α vector.

    Notes
    -----
    ``near_zero_signal`` is set when ‖x_syn‖ < _NEAR_ZERO_THRESHOLD; in that
    case NRMSE = ‖ε‖ / ‖x_syn‖ is numerically unreliable and should be
    excluded from aggregate statistics.
    """
    n_real = X_real_scaled.shape[0]

    def objective(alpha: np.ndarray) -> float:
        x_approx = X_real_scaled.T @ alpha
        reconstruction_loss = np.sum((x_approx - x_syn_scaled) ** 2)
        entropy = -np.sum(alpha * np.log(alpha + 1e-12))
        return reconstruction_loss + lam_sparsity * entropy

    def gradient(alpha: np.ndarray) -> np.ndarray:
        x_approx = X_real_scaled.T @ alpha
        grad_reconstruction = 2.0 * X_real_scaled @ (x_approx - x_syn_scaled)
        grad_entropy = -(np.log(alpha + 1e-12) + 1.0)
        return grad_reconstruction + lam_sparsity * grad_entropy

    alpha0 = np.ones(n_real) / n_real

    if convex:
        constraints = {"type": "eq", "fun": lambda a: a.sum() - 1.0}
        bounds = [(0.0, 1.0)] * n_real
        result = minimize(
            objective, alpha0, jac=gradient, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        alpha = np.clip(result.x, 0.0, None)
        alpha /= alpha.sum() + 1e-12
    else:
        bounds = [(0.0, None)] * n_real
        result = minimize(
            objective, alpha0, jac=gradient, method="L-BFGS-B",
            bounds=bounds, options={"ftol": 1e-12, "maxiter": 500},
        )
        alpha = np.clip(result.x, 0.0, None)

    x_approx = X_real_scaled.T @ alpha
    residual = x_syn_scaled - x_approx

    noise_norm = np.linalg.norm(residual)
    signal_norm = np.linalg.norm(x_syn_scaled)
    near_zero_signal = signal_norm < _NEAR_ZERO_THRESHOLD
    nrmse = noise_norm / (signal_norm + 1e-12)  # unreliable when near_zero_signal

    approx_norm = np.linalg.norm(x_approx)
    cos_sim = float(
        (x_syn_scaled @ x_approx) / (signal_norm * approx_norm + 1e-12)
    )

    return {
        "alpha": alpha,
        "x_approx_scaled": x_approx,
        "residual_scaled": residual,
        "nrmse": nrmse,
        "near_zero_signal": near_zero_signal,
        "cos_sim": cos_sim,
        "noise_norm_scaled": noise_norm,
        "alpha_sum": alpha.sum(),
        "alpha_max": alpha.max(),
        "alpha_entropy": -np.sum(alpha * np.log(alpha + 1e-12)),
        "n_dominant": int((alpha > _DOMINANT_THRESHOLD).sum()),
        "converged": result.success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Single-run analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_run(
    X_real_scaled: np.ndarray,
    X_syn_scaled: np.ndarray,
    X_real_orig: np.ndarray,
    X_syn_orig: np.ndarray,
    lam_sparsity: float = 0.05,
    convex: bool = True,
) -> pd.DataFrame:
    """
    Decompose every synthetic row and collect per-row metrics.

    Metrics are computed both in the scaled (StandardScaler) space and in the
    original (unscaled) space so callers can compare both.

    Parameters
    ----------
    X_real_scaled, X_syn_scaled : scaled arrays (n_real/n_syn × n_features)
    X_real_orig,   X_syn_orig   : original unscaled counterparts
    lam_sparsity, convex        : passed through to fit_combination_regularized

    Returns
    -------
    pd.DataFrame indexed by syn_idx, with columns for NRMSE, cosine similarity,
    α statistics, per-feature residuals (resid_0 … resid_F), and α weights
    (alpha_0 … alpha_R).
    """
    n_syn = X_syn_scaled.shape[0]
    n_real = X_real_scaled.shape[0]
    records = []

    for i in range(n_syn):
        fit = fit_combination_regularized(
            X_syn_scaled[i], X_real_scaled, lam_sparsity, convex
        )

        # Reconstruction metrics in original (unscaled) space.
        x_approx_orig = X_real_orig.T @ fit["alpha"]
        x_orig = X_syn_orig[i]
        residual_orig = x_orig - x_approx_orig
        noise_norm_orig = np.linalg.norm(residual_orig)
        signal_norm_orig = np.linalg.norm(x_orig)
        nrmse_orig = noise_norm_orig / (signal_norm_orig + 1e-12)
        cos_sim_orig = float(
            (x_orig @ x_approx_orig)
            / (signal_norm_orig * np.linalg.norm(x_approx_orig) + 1e-12)
        )

        record: dict = {
            "syn_idx": i,
            # Scaled-space metrics.
            "nrmse": fit["nrmse"],
            "near_zero_signal": fit["near_zero_signal"],
            "cos_sim": fit["cos_sim"],
            # Original-space metrics.
            "nrmse_orig": nrmse_orig,
            "cos_sim_orig": cos_sim_orig,
            # α statistics.
            "noise_norm_scaled": fit["noise_norm_scaled"],
            "alpha_sum": fit["alpha_sum"],
            "alpha_max": fit["alpha_max"],
            "alpha_entropy": fit["alpha_entropy"],
            "n_dominant": fit["n_dominant"],
            "converged": fit["converged"],
        }

        # Store per-feature residuals for downstream noise analysis.
        for f, v in enumerate(fit["residual_scaled"]):
            record[f"resid_{f}"] = v

        # Store individual α weights for heatmap visualization.
        for j in range(n_real):
            record[f"alpha_{j}"] = fit["alpha"][j]

        records.append(record)

    return pd.DataFrame(records).set_index("syn_idx")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-run aggregation
# ─────────────────────────────────────────────────────────────────────────────

def analyze_all_runs(
    X_real: pd.DataFrame,
    X_syn_list: list[pd.DataFrame],
    lam_sparsity: float = 0.05,
    convex: bool = True,
) -> dict:
    """
    Fit and aggregate decompositions across multiple LLM generation runs.

    A single StandardScaler is fitted on X_real and applied to all X_syn
    matrices, ensuring a consistent feature space across runs.

    Parameters
    ----------
    X_real      : real data, shape (n_real, n_features)
    X_syn_list  : list of synthetic DataFrames, each shape (n_syn, n_features)
    lam_sparsity, convex : passed through to analyze_run

    Returns
    -------
    dict with keys: full_df, scaler, diag, feature_names,
    and aggregated scalar metrics (nrmse_mean, cos_sim_mean, …).
    """
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real.values.astype(float))
    X_real_orig = X_real.values.astype(float)

    diag = check_overdetermination(X_real_scaled)
    _print_basis_diagnostics(diag, lam_sparsity)

    run_dfs = []
    for run_idx, X_syn in enumerate(X_syn_list):
        X_syn_scaled = scaler.transform(X_syn.values.astype(float))
        X_syn_orig = X_syn.values.astype(float)
        df = analyze_run(
            X_real_scaled, X_syn_scaled,
            X_real_orig, X_syn_orig,
            lam_sparsity, convex,
        )
        df["run"] = run_idx
        run_dfs.append(df)

    full_df = pd.concat(run_dfs).reset_index()

    n_near_zero = full_df["near_zero_signal"].sum()
    if n_near_zero > 0:
        print(
            f"    ⚠  {n_near_zero} rows with near_zero_signal: "
            "NRMSE excluded from mean/std."
        )

    alpha_cols = [
        c for c in full_df.columns
        if c.startswith("alpha_")
        and c not in ("alpha_sum", "alpha_max", "alpha_entropy")
    ]

    # NRMSE statistics exclude near-zero-signal rows (unreliable denominator).
    valid_nrmse = full_df.loc[~full_df["near_zero_signal"], "nrmse"]

    return {
        "full_df": full_df,
        "scaler": scaler,
        "diag": diag,
        "feature_names": X_real.columns.tolist(),
        # Scaled-space metrics (near-zero rows excluded from NRMSE).
        "nrmse_mean": valid_nrmse.mean(),
        "nrmse_std": valid_nrmse.std(),
        "nrmse_median": valid_nrmse.median(),
        "cos_sim_mean": full_df["cos_sim"].mean(),
        "cos_sim_std": full_df["cos_sim"].std(),
        # Original-space metrics.
        "nrmse_orig_mean": full_df["nrmse_orig"].mean(),
        "nrmse_orig_std": full_df["nrmse_orig"].std(),
        "cos_sim_orig_mean": full_df["cos_sim_orig"].mean(),
        # Noise norm.
        "noise_norm_mean": full_df["noise_norm_scaled"].mean(),
        "noise_norm_std": full_df["noise_norm_scaled"].std(),
        # Alpha statistics.
        "n_dominant_mean": full_df["n_dominant"].mean(),
        "mean_alpha_matrix": full_df.groupby("syn_idx")[alpha_cols].mean(),
        "std_alpha_matrix": full_df.groupby("syn_idx")[alpha_cols].std(),
    }


def _print_basis_diagnostics(diag: dict, lam_sparsity: float) -> None:
    """Print a concise basis-health report to stdout."""
    print("\n  Basis diagnostics:")
    print(f"    Features : {diag['n_features']}  |  Real rows : {diag['n_real']}")
    print(f"    Matrix rank      : {diag['rank']}")
    print(f"    Condition number : {diag['condition_number']:.1f}")

    if diag["is_underdetermined"]:
        print(
            f"    ⚠  Underdetermined: n_features ({diag['n_features']}) "
            f"<= n_real ({diag['n_real']})"
        )
        print(
            f"       NRMSE ≡ 0 trivially at λ=0 → "
            f"using regularization λ={lam_sparsity}"
        )
        if lam_sparsity == 0:
            print(
                "    ✗  lam_sparsity=0 in underdetermined regime — results are meaningless!\n"
                "       Set lam_sparsity in the range 0.01–0.2."
            )
    else:
        print(
            f"    ✓  Overdetermined: n_features > n_real — "
            "problem is well-posed."
        )
        if lam_sparsity > 0.1:
            print(
                f"    ⚠  lam_sparsity={lam_sparsity} is high for the overdetermined case.\n"
                "       Recommended range: 0.01–0.05 (higher values over-sparsify α)."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Residual noise structure tests
# ─────────────────────────────────────────────────────────────────────────────

def test_noise_structure(
    full_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Analyze the statistical structure of per-feature residuals.

    Residuals are read from columns ``resid_0``, ``resid_1``, … that were
    stored by :func:`analyze_run`; no re-optimization is performed.

    Per-feature tests
    -----------------
    * **Normality** – Shapiro–Wilk (n ≤ 5 000) and D'Agostino–Pearson
      (n ≥ 8).  ``is_normal`` is based on the D'Agostino p-value.
    * **Bias** – one-sample t-test (H₀: mean residual = 0).
      ``has_bias`` is True when p < 0.05.
    * **bias_ratio** = |mean| / std — supplementary scale-free bias measure.
    * **skewness** — asymmetry of the residual distribution.

    Parameters
    ----------
    full_df       : aggregated DataFrame from analyze_all_runs
    feature_names : ordered list of feature names

    Returns
    -------
    pd.DataFrame, one row per feature.
    """
    records = []

    for f, name in enumerate(feature_names):
        col = f"resid_{f}"
        if col not in full_df.columns:
            continue

        residuals = full_df[col].dropna().values

        # Normality tests.
        sw_p = shapiro(residuals)[1] if len(residuals) <= 5_000 else np.nan
        da_p = normaltest(residuals)[1] if len(residuals) >= 8 else np.nan

        # Bias test: H₀: E[residual] = 0.
        t_p = ttest_1samp(residuals, popmean=0)[1] if len(residuals) >= 2 else np.nan

        bias_ratio = abs(residuals.mean()) / (residuals.std() + 1e-12)

        records.append({
            "feature": name,
            "residual_mean_norm": residuals.mean(),
            "residual_std_norm": residuals.std(),
            "bias_ratio": bias_ratio,
            "ttest_p": t_p,
            "has_bias": (t_p < _ALPHA_SIGNIFICANCE) if not np.isnan(t_p) else False,
            "skewness": skew(residuals),
            "shapiro_p": sw_p,
            "dagostino_p": da_p,
            "is_normal": (da_p > _ALPHA_SIGNIFICANCE) if not np.isnan(da_p) else True,
            "n_obs": len(residuals),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Diagnostic figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_analysis(
    summary: dict,
    noise_df: pd.DataFrame,
    convex: bool = True,
    lam_sparsity: float = 0.05,
    save_path: str | None = None,
) -> None:
    """
    Render a 6-panel diagnostic figure (2 × 3 grid).

    Panels
    ------
    (A) NRMSE distribution histogram
    (B) Number of dominant real examples per synthetic row (bar + error)
    (C) Mean α weight heatmap (synthetic rows × real examples)
    (D) Std α heatmap — stability across runs
    (E) Structural bias per feature (−log₁₀ p-value, t-test)
    (F) Residual normality per feature (−log₁₀ p-value, D'Agostino)

    Parameters
    ----------
    summary      : dict returned by analyze_all_runs
    noise_df     : DataFrame returned by test_noise_structure
    convex       : used for the (C) panel title label
    lam_sparsity : used in (A) panel title
    save_path    : if provided, save the figure to this path at 150 dpi
    """
    full_df = summary["full_df"]
    mean_alpha = summary["mean_alpha_matrix"]
    std_alpha = summary["std_alpha_matrix"]
    n_real = mean_alpha.shape[1]

    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── (A) NRMSE distribution ───────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(
        full_df["nrmse"].dropna(), bins=25,
        color="#4A90D9", edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    ax_a.axvline(
        summary["nrmse_mean"], color="#E05B3C", linewidth=1.8,
        linestyle="--", label=f'mean = {summary["nrmse_mean"]:.3f}',
    )
    ax_a.axvline(
        _NRMSE_THRESHOLD, color="#2CA05A", linewidth=1.2,
        linestyle=":", label=f"threshold {_NRMSE_THRESHOLD}",
    )
    ax_a.set_xlabel("NRMSE = ‖ε‖ / ‖x_syn‖  (scaled space)", fontsize=10)
    ax_a.set_ylabel("Count", fontsize=10)
    ax_a.set_title(
        f"(A) NRMSE: residual norm / signal norm\n"
        f"0 = perfect fit  |  λ_sparsity = {lam_sparsity}",
        fontsize=11, fontweight="bold",
    )
    ax_a.legend(fontsize=8)
    ax_a.spines[["top", "right"]].set_visible(False)

    # ── (B) Dominant real examples per synthetic row ─────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    dom_mean = full_df.groupby("syn_idx")["n_dominant"].mean()
    dom_std = full_df.groupby("syn_idx")["n_dominant"].std().fillna(0)
    ax_b.bar(
        dom_mean.index, dom_mean.values, yerr=dom_std.values,
        color="#9B6AC4", alpha=0.85, edgecolor="white",
        capsize=4, linewidth=0.5,
    )
    ax_b.set_xlabel("Synthetic row index", fontsize=10)
    ax_b.set_ylabel(f"N real examples with α > {_DOMINANT_THRESHOLD}", fontsize=10)
    ax_b.set_title(
        "(B) Effective number of dominant\nreal examples (mean ± std)",
        fontsize=11, fontweight="bold",
    )
    ax_b.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax_b.spines[["top", "right"]].set_visible(False)

    # ── (C) Mean α heatmap ───────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    im = ax_c.imshow(
        mean_alpha.values, aspect="auto", cmap="YlOrRd",
        interpolation="nearest", vmin=0,
    )
    plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04, label="mean α")
    ax_c.set_xticks(range(n_real))
    ax_c.set_xticklabels(
        [f"r{j}" for j in range(n_real)], rotation=45, ha="right", fontsize=7,
    )
    ax_c.set_yticks(range(len(mean_alpha)))
    ax_c.set_yticklabels([f"s{i}" for i in mean_alpha.index], fontsize=7)
    ax_c.set_xlabel("Real example", fontsize=10)
    ax_c.set_ylabel("Synthetic row", fontsize=10)
    kind_label = "convex" if convex else "NNLS"
    ax_c.set_title(f"(C) Mean α weights ({kind_label})", fontsize=11, fontweight="bold")

    # ── (D) Std α — stability across runs ───────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    im2 = ax_d.imshow(
        std_alpha.values, aspect="auto", cmap="Blues",
        interpolation="nearest", vmin=0,
    )
    plt.colorbar(im2, ax=ax_d, fraction=0.046, pad=0.04, label="std α")
    ax_d.set_xticks(range(n_real))
    ax_d.set_xticklabels(
        [f"r{j}" for j in range(n_real)], rotation=45, ha="right", fontsize=7,
    )
    ax_d.set_yticks(range(len(std_alpha)))
    ax_d.set_yticklabels([f"s{i}" for i in std_alpha.index], fontsize=7)
    ax_d.set_xlabel("Real example", fontsize=10)
    ax_d.set_ylabel("Synthetic row", fontsize=10)
    ax_d.set_title(
        "  (D) α stability across runs\n(std — lower is more stable)",
        fontsize=11, fontweight="bold",
    )

    # ── (E) Structural bias per feature ─────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    sorted_bias = noise_df.sort_values("ttest_p", ascending=False)
    colors_e = [
        "#E05B3C" if has_bias else "#4A90D9"
        for has_bias in sorted_bias["has_bias"]
    ]
    ax_e.barh(
        sorted_bias["feature"],
        -np.log10(sorted_bias["ttest_p"].clip(lower=1e-12)),
        color=colors_e, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax_e.axvline(
        -np.log10(_ALPHA_SIGNIFICANCE), color="gray",
        linewidth=1.2, linestyle="--",
    )
    ax_e.set_xlabel(f"−log₁₀(p-value)  [t-test: mean residual = 0]", fontsize=10)
    ax_e.set_title(
        "(E) Structural bias by feature\nred = t-test rejected (p < 0.05)",
        fontsize=11, fontweight="bold",
    )
    ax_e.legend(
        handles=[
            Patch(facecolor="#E05B3C", label="Bias (p < 0.05)"),
            Patch(facecolor="#4A90D9", label="No bias"),
            plt.Line2D([0], [0], color="gray", linestyle="--", label="p = 0.05"),
        ],
        fontsize=8,
    )
    ax_e.spines[["top", "right"]].set_visible(False)

    # ── (F) Residual normality per feature ───────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    sorted_norm = noise_df.sort_values("dagostino_p", ascending=False)
    colors_f = [
        "#2CA05A" if is_normal else "#E05B3C"
        for is_normal in sorted_norm["is_normal"]
    ]
    ax_f.barh(
        sorted_norm["feature"],
        -np.log10(sorted_norm["dagostino_p"] + 1e-12),
        color=colors_f, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax_f.axvline(
        -np.log10(_ALPHA_SIGNIFICANCE), color="gray",
        linewidth=1.2, linestyle="--",
    )
    ax_f.set_xlabel("−log₁₀(p-value) [D'Agostino]", fontsize=10)
    ax_f.set_title(
        "(F) Residual normality by feature\ngreen = Gaussian noise, red = structured",
        fontsize=11, fontweight="bold",
    )
    ax_f.legend(
        handles=[
            Patch(facecolor="#2CA05A", label="Normal (p > 0.05)"),
            Patch(facecolor="#E05B3C", label="Non-normal"),
            plt.Line2D([0], [0], color="gray", linestyle="--", label="p = 0.05"),
        ],
        fontsize=8,
    )
    ax_f.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        f"Synthetic decomposition  |  "
        f"NRMSE (scaled) = {summary['nrmse_mean']:.3f} ± {summary['nrmse_std']:.3f}  |  "
        f"cos_sim = {summary['cos_sim_mean']:.3f}  |  "
        f"N_dominant mean = {summary['n_dominant_mean']:.1f}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Figure saved: {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. End-to-end entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(
    X_real: pd.DataFrame,
    X_syn_list: list[pd.DataFrame],
    convex: bool = True,
    lam_sparsity: float = 0.05,
    save_plot: str | None = "analysis_plot.png",
) -> dict:
    """
    Run the complete synthetic decomposition pipeline.

    Parameters
    ----------
    X_real        : real data, shape (n_real, n_features)
    X_syn_list    : list of synthetic DataFrames from separate LLM runs
    convex        : True → convex combination (Σα = 1, α ≥ 0) via SLSQP;
                    False → non-negative combination (α ≥ 0) via L-BFGS-B
    lam_sparsity  : sparsity penalty for uniform α distributions.
                    0 = pure NNLS / convex QP (may give trivial R² ≡ 1
                    in underdetermined regime).
                    Recommended: 0.01–0.1; higher → sparser α.
    save_plot     : file path for the output figure; None to skip saving.

    Returns
    -------
    dict with keys ``summary`` and ``noise_df``.
    """
    combination_type = "convex" if convex else "NNLS"
    print("=" * 60)
    print(f"  Synthetic decomposition  |  {combination_type}  |  λ_sparsity={lam_sparsity}")
    print(f"  Real rows : {len(X_real)}  |  Runs : {len(X_syn_list)}")
    print("=" * 60)

    print("\n[1/4] Decomposition + basis diagnostics …")
    summary = analyze_all_runs(X_real, X_syn_list, lam_sparsity=lam_sparsity, convex=convex)

    print("\n[2/4] Noise structure analysis …")
    noise_df = test_noise_structure(summary["full_df"], summary["feature_names"])

    print("\n[3/4] Results:")
    print(f"  NRMSE scaled   mean ± std  : {summary['nrmse_mean']:.4f} ± {summary['nrmse_std']:.4f}")
    print(f"  NRMSE scaled   median      : {summary['nrmse_median']:.4f}")
    print(f"  Cosine sim     mean ± std  : {summary['cos_sim_mean']:.4f} ± {summary['cos_sim_std']:.4f}")
    print(f"  NRMSE original mean ± std  : {summary['nrmse_orig_mean']:.4f} ± {summary['nrmse_orig_std']:.4f}")
    print(f"  Cosine sim original mean   : {summary['cos_sim_orig_mean']:.4f}")
    print(f"  Noise ‖ε‖ scaled mean ± std: {summary['noise_norm_mean']:.4f} ± {summary['noise_norm_std']:.4f}")
    print(f"  N dominant mean            : {summary['n_dominant_mean']:.2f}  (of {len(X_real)} real rows)")

    n_biased = noise_df["has_bias"].sum()
    n_normal = noise_df["is_normal"].sum()
    print(f"\n  Normal residuals   : {n_normal}/{len(noise_df)} features")
    print(f"  Structural bias    : {n_biased}/{len(noise_df)} features  (t-test p < 0.05)")

    if n_biased > 0:
        print("\n  Features with significant bias (t-test on mean = 0 rejected):")
        biased_features = noise_df[noise_df["has_bias"]].sort_values("ttest_p")
        print(
            biased_features[
                ["feature", "residual_mean_norm", "residual_std_norm",
                 "bias_ratio", "ttest_p", "skewness"]
            ].to_string(index=False)
        )

    nrmse_ok = summary["nrmse_mean"] < _NRMSE_THRESHOLD
    cos_ok = summary["cos_sim_mean"] > _COSINE_THRESHOLD
    verdict = "✓" if (nrmse_ok and cos_ok) else "⚠"
    print(
        f"\n  {verdict}  Hypothesis: "
        f"NRMSE < {_NRMSE_THRESHOLD} {'✓' if nrmse_ok else '✗'}  |  "
        f"cos_sim > {_COSINE_THRESHOLD} {'✓' if cos_ok else '✗'}"
    )

    print("\n[4/4] Plotting …")
    plot_analysis(summary, noise_df, convex=convex, lam_sparsity=lam_sparsity, save_path=save_plot)

    return {"summary": summary, "noise_df": noise_df}
