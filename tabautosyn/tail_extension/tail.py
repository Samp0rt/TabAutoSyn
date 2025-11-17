import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy, wasserstein_distance
import matplotlib.pyplot as plt
from Matrix import matrix_eigenvalue_mse, matrix_cosine_similarity, matrix_frobenius_distance


# ==========================================================
# DISTANCE METRICS
# ==========================================================

def robust_mahalanobis_distances(X, center=None, cov=None):
    """Compute robust Mahalanobis distances for each sample in X."""
    X = np.asarray(X)
    if center is None:
        center = np.mean(X, axis=0)
    if cov is None:
        cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    diff = X - center
    sq = np.sum((diff @ cov_inv) * diff, axis=1)
    return np.sqrt(np.maximum(sq, 0.0))


def energy_distance(X, Y):
    """Compute energy distance between two datasets."""
    Dxy = pairwise_distances(X, Y)
    Dxx = pairwise_distances(X, X)
    Dyy = pairwise_distances(Y, Y)
    val = 2 * np.mean(Dxy) - np.mean(Dxx) - np.mean(Dyy)
    return np.sqrt(max(val, 0.0))


# ==========================================================
# DIVERGENCE METRICS
# ==========================================================

def kde_js_divergence(X, Y, bandwidth=0.2, grid_size=800, random_state=0):
    """Compute Jensen-Shannon divergence using KDE approximation."""
    rng = np.random.default_rng(random_state)
    X_comb = np.vstack([X, Y])
    mins, maxs = X_comb.min(axis=0), X_comb.max(axis=0)
    samples = rng.uniform(mins, maxs, size=(grid_size, X.shape[1]))
    
    kde_X = KernelDensity(bandwidth=bandwidth).fit(X)
    kde_Y = KernelDensity(bandwidth=bandwidth).fit(Y)
    log_px, log_py = kde_X.score_samples(samples), kde_Y.score_samples(samples)
    px, py = np.exp(log_px), np.exp(log_py)
    m = 0.5 * (px + py)
    
    js = 0.5 * (np.sum(px * np.log((px + 1e-12) / (m + 1e-12))) +
                np.sum(py * np.log((py + 1e-12) / (m + 1e-12))))
    js /= len(px)
    return float(js)


def js_divergence_fast(X, Y, method="hist", bins=50):
    """Fast JS divergence approximation using histograms or entropy."""
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    
    xmin = min(X.min(), Y.min())
    xmax = max(X.max(), Y.max())
    bins_edges = np.linspace(xmin, xmax, bins + 1)
    
    px, _ = np.histogram(X, bins=bins_edges, density=True)
    py, _ = np.histogram(Y, bins=bins_edges, density=True)
    px = px / (px.sum() + 1e-12)
    py = py / (py.sum() + 1e-12)
    m = 0.5 * (px + py)
    
    if method == "hist":
        js = 0.5 * (np.sum(px * np.log((px + 1e-12) / (m + 1e-12))) +
                    np.sum(py * np.log((py + 1e-12) / (m + 1e-12))))
        return float(np.sqrt(max(js, 0.0)))
    
    elif method == "entropy":
        js = 0.5 * (entropy(px, m) + entropy(py, m))
        return float(np.sqrt(max(js, 0.0)))
    
    elif method == "wasserstein":
        return float(wasserstein_distance(X, Y))
    
    else:
        raise ValueError("method must be 'hist', 'entropy', or 'wasserstein'")


# ==========================================================
# CLUSTERING
# ==========================================================

def cluster_tail_candidates(X_tail, n_clusters=10, random_state=0):
    """Cluster tail candidates and return representative indices."""
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X_tail)
    centers = model.cluster_centers_
    
    reps = []
    for i in range(n_clusters):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        dists = np.linalg.norm(X_tail[idx] - centers[i], axis=1)
        reps.append(idx[np.argmin(dists)])
    return np.array(reps)


# ==========================================================
# DISTANCE FUNCTION 
# ==========================================================

def create_distance_function(metric, center, cov):
    """Factory to create distance function based on metric type."""
    if metric == "mahalanobis":
        return lambda X: robust_mahalanobis_distances(X, center, cov)
    
    elif metric == "cosine":
        def dist_func(X):
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            cn = center / (np.linalg.norm(center) + 1e-12)
            return 1 - np.dot(Xn, cn)
        return dist_func
    
    elif metric == "correlation":
        def dist_func(X):
            Xc = X - X.mean(axis=1, keepdims=True)
            cc = center - np.mean(center)
            Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
            cn = cc / (np.linalg.norm(cc) + 1e-12)
            return 1 - np.dot(Xn, cn)
        return dist_func
    
    else:
        raise ValueError("distance_metric must be 'mahalanobis', 'cosine', or 'correlation'")


# ==========================================================
# DIVERGENCE COMPUTATION
# ==========================================================

def compute_weighted_divergence(Xa, Xb, cols, weights, divergence_metric, 
                                divergence_variant, compare_metric, random_state):
    """Compute weighted divergence between two distributions."""
    # Matrix comparison metrics (no weighting needed)
    if divergence_metric == "matrix_eigenvalue_mse":
        _, _, _, _, mse = matrix_eigenvalue_mse(
            pd.DataFrame(Xa, columns=cols),
            pd.DataFrame(Xb, columns=cols),
            metric=compare_metric,
            show_plots=False
        )
        return mse
    
    elif divergence_metric == "matrix_cosine_similarity":
        result = matrix_cosine_similarity(
            pd.DataFrame(Xa, columns=cols),
            pd.DataFrame(Xb, columns=cols),metric=compare_metric
        )
        # Handle both single value and tuple returns
        cos_sim = result[2] if isinstance(result, tuple) else result
        return 1 - cos_sim  # Convert similarity to distance
    
    elif divergence_metric == "matrix_frobenius_distance":
        result = matrix_frobenius_distance(
            pd.DataFrame(Xa, columns=cols),
            pd.DataFrame(Xb, columns=cols),metric=compare_metric
        )
        # Handle both single value and tuple returns
        frob_dist = result[3] if isinstance(result, tuple) else result
        return frob_dist
    
    # Feature-wise metrics (with weighting)
    total = 0.0
    for j in range(Xa.shape[1]):
        if divergence_metric == "js":
            if divergence_variant == "kde":
                d_j = kde_js_divergence(Xa[:, [j]], Xb[:, [j]], random_state=random_state)
            else:
                d_j = js_divergence_fast(Xa[:, [j]], Xb[:, [j]], method=divergence_variant)
        
        elif divergence_metric == "energy":
            d_j = energy_distance(Xa[:, [j]], Xb[:, [j]])
        
        elif divergence_metric == "wasserstein":
            d_j = wasserstein_distance(Xa[:, j], Xb[:, j])
        
        else:
            raise ValueError(
                "divergence_metric must be 'js', 'energy', 'wasserstein', "
                "'matrix_eigenvalue_mse', 'matrix_cosine_similarity', or 'matrix_frobenius_distance'"
            )
        
        total += weights[j] * d_j
    
    return total


# ==========================================================
# MAIN CORRECTION FUNCTION
# ==========================================================

def correct_tails_by_adding(
    df_real,
    df_syn,
    df_syn_tail,
    tail_quantile=0.95,
    divergence_metric="matrix_eigenvalue_mse",
    divergence_variant="hist",
    compare_metric="mi",
    distance_metric="mahalanobis",
    search_strategy="stochastic",
    loss_scope="global",
    use_clustering=False,
    n_tail_clusters=1000,
    max_additions=10000,
    stochastic_trials=200,
    hybrid_alpha=1.0,
    improvement_threshold=1e-4,
    random_state=0,
    verbose=True,
    scaler=False
):
    """
    Correct synthetic data by adding tail samples iteratively.
    
    Parameters:
    -----------
    df_real : pd.DataFrame
        Real data
    df_syn : pd.DataFrame
        Synthetic data to be corrected
    df_syn_tail : pd.DataFrame
        Pool of tail candidates for addition
    tail_quantile : float
        Quantile threshold for tail definition
    divergence_metric : str
        Metric for distribution comparison: 'js', 'energy', 'wasserstein', 
        'matrix_real_synth', 'matrix_cosine_similarity', 'matrix_frobenius_distance'
    divergence_variant : str
        Variant for JS divergence: 'kde', 'hist', 'entropy', 'wasserstein'
    compare_metric : str
        Metric for compare_real_synth: 'mi', 'wasserstein', 'js'
    distance_metric : str
        Distance metric for tail detection: 'mahalanobis', 'cosine', 'correlation'
    search_strategy : str
        Search strategy: 'stochastic' or 'greedy'
    loss_scope : str
        Loss computation scope: 'hybrid' (tail+center) or 'global' (full distribution)
    use_clustering : bool
        Whether to cluster tail candidates
    n_tail_clusters : int
        Number of clusters for tail candidates
    max_additions : int
        Maximum number of samples to add
    stochastic_trials : int
        Number of trials per iteration for stochastic search
    hybrid_alpha : float
        Weight for tail divergence in hybrid loss
    improvement_threshold : float
        Minimum improvement threshold
    random_state : int
        Random seed
    verbose : bool
        Whether to print progress
    scaler : bool
        Whether to use RobustScaler for normalization
    
    Returns:
    --------
    df_syn_corrected : pd.DataFrame
        Corrected synthetic data
    df_added : pd.DataFrame
        Added samples
    history_df : pd.DataFrame
        Optimization history
    """
    rng = np.random.default_rng(random_state)
    cols = df_real.columns.tolist()
    
    # === Normalization ===
    if scaler:
        scaler_obj = RobustScaler()
        Xr = scaler_obj.fit_transform(df_real.values)
        Xs = scaler_obj.transform(df_syn.values)
        Xt = scaler_obj.transform(df_syn_tail.values)
    else:
        Xr = df_real.values
        Xs = df_syn.values
        Xt = df_syn_tail.values
    
    # === Distance computation setup ===
    center = np.mean(Xr, axis=0)
    cov = np.cov(Xr, rowvar=False)
    dist_func = create_distance_function(distance_metric, center, cov)
    
    d_real = dist_func(Xr)
    #d_syn = dist_func(Xs)
    #d_tail = dist_func(Xt)
    thr = np.quantile(d_real, tail_quantile)
    
    tail_real = Xr[d_real >= thr]
    center_real = Xr[d_real < thr]
    tail_candidates = Xt
    
    # === Clustering (optional) ===
    if use_clustering and len(tail_candidates) > n_tail_clusters:
        reps_idx = cluster_tail_candidates(tail_candidates, n_tail_clusters, random_state)
        tail_candidates = tail_candidates[reps_idx]
        if verbose:
            print(f"[+] Clustered tail: selected {len(tail_candidates)} representatives")
    else:
        if verbose:
            print(f"[+] Using {len(tail_candidates)} tail candidates")
    
    # === Feature weights ===
    sigmas = np.std(Xr, axis=0)
    weights = 1 / (sigmas + 1e-8)
    weights /= np.sum(weights)
    
    # === Objective function ===
    def compute_objective(X_syn):
        if loss_scope == "global":
            J = compute_weighted_divergence(
                Xr, X_syn, cols, weights, divergence_metric, 
                divergence_variant, compare_metric, random_state
            )
            return J, np.nan, np.nan
        else:  # hybrid
            d_temp = dist_func(X_syn)
            tail_syn = X_syn[d_temp >= thr]
            center_syn = X_syn[d_temp < thr]
            
            D_tail = compute_weighted_divergence(
                tail_real, tail_syn, cols, weights, divergence_metric,
                divergence_variant, compare_metric, random_state
            )
            D_center = compute_weighted_divergence(
                center_real, center_syn, cols, weights, divergence_metric,
                divergence_variant, compare_metric, random_state
            )
            J = hybrid_alpha * D_tail + (1 - hybrid_alpha) * D_center
            return J, D_tail, D_center
    
    # === Initial state ===
    J, D_tail, D_center = compute_objective(Xs)
    if verbose:
        if loss_scope == "global":
            print(f"[i] Initial global loss: J={J:.6f}")
        else:
            print(f"[i] Initial hybrid objective: J={J:.6f}, D_tail={D_tail:.6f}, D_center={D_center:.6f}")
    
    history = [(0, J, D_tail, D_center)]
    added_samples = []
    added = 0
    improved = True
    used_indices = set()
    
    # === Optimization loop ===
    while improved and added < max_additions:
        improved = False
        
        if search_strategy == "stochastic":
            for _ in range(stochastic_trials):
                idx = rng.integers(0, len(tail_candidates))
                if idx in used_indices:
                    continue
                
                cand = tail_candidates[idx]
                X_temp = np.vstack([Xs, cand])
                J_temp, Dt_temp, Dc_temp = compute_objective(X_temp)
                
                if J_temp < J - improvement_threshold:
                    Xs = X_temp
                    J, D_tail, D_center = J_temp, Dt_temp, Dc_temp
                    added_samples.append(cand)
                    added += 1
                    used_indices.add(idx)
                    history.append((added, J, D_tail, D_center))
                    improved = True
                    
                    if verbose:
                        if loss_scope == "global":
                            print(f"[{added}] Improvement: J={J:.6f}")
                        else:
                            print(f"[{added}] Improvement: J={J:.6f}, D_tail={D_tail:.6f}, D_center={D_center:.6f}")
                    break
        
        elif search_strategy == "greedy":
            best_local = J
            best_cand = None
            best_metrics = None
            best_idx = None
            
            for idx, cand in enumerate(tail_candidates):
                if idx in used_indices:
                    continue
                
                X_temp = np.vstack([Xs, cand])
                J_temp, Dt_temp, Dc_temp = compute_objective(X_temp)
                
                if J_temp < best_local - improvement_threshold:
                    best_local = J_temp
                    best_cand = cand
                    best_metrics = (Dt_temp, Dc_temp)
                    best_idx = idx
            
            if best_cand is not None:
                Xs = np.vstack([Xs, best_cand])
                J = best_local
                if loss_scope != "global":
                    D_tail, D_center = best_metrics
                added_samples.append(best_cand)
                added += 1
                used_indices.add(best_idx)
                history.append((added, J, D_tail, D_center))
                improved = True
                
                if verbose:
                    if loss_scope == "global":
                        print(f"[{added}] Improvement (greedy): J={J:.6f}")
                    else:
                        print(f"[{added}] Improvement (greedy): J={J:.6f}, D_tail={D_tail:.6f}, D_center={D_center:.6f}")
    
    # === Output ===
    if verbose:
        if loss_scope == "global":
            print(f"\n[✓] Final global loss: J={J:.6f}")
        else:
            print(f"\n[✓] Final values: J={J:.6f}, D_tail={D_tail:.6f}, D_center={D_center:.6f}")
        print(f"[✓] Added {added} tail samples (α={hybrid_alpha})")
    
    # Inverse transform if scaler was used
    if scaler:
        Xs_unscaled = scaler_obj.inverse_transform(Xs)
        added_unscaled = scaler_obj.inverse_transform(np.vstack(added_samples)) if added_samples else np.empty((0, Xr.shape[1]))
    else:
        Xs_unscaled = Xs
        added_unscaled = np.vstack(added_samples) if added_samples else np.empty((0, Xr.shape[1]))
    
    hist_df = pd.DataFrame(history, columns=["step", "J", "D_tail", "D_center"])
    
    return pd.DataFrame(Xs_unscaled, columns=cols), pd.DataFrame(added_unscaled, columns=cols), hist_df


# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_hybrid_metrics(history_df):
    """Plot optimization history."""
    plt.figure(figsize=(7, 4))
    plt.plot(history_df["step"], history_df["J"], label="Hybrid J", lw=2)
    plt.plot(history_df["step"], history_df["D_tail"], "--", label="Tail Divergence")
    plt.plot(history_df["step"], history_df["D_center"], "--", label="Center Divergence")
    plt.xlabel("Iteration")
    plt.ylabel("Divergence")
    plt.title("Dynamics of hybrid objective J")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_tail_distributions(df_real, df_syn_before, df_syn_after, tail_quantile=0.95):
    """Plot Mahalanobis distance distributions before and after correction."""
    scaler_obj = RobustScaler()
    Xr = scaler_obj.fit_transform(df_real.values)
    Xb = scaler_obj.transform(df_syn_before.values)
    Xa = scaler_obj.transform(df_syn_after.values)
    
    center = np.mean(Xr, axis=0)
    cov = np.cov(Xr, rowvar=False)
    
    d_real = robust_mahalanobis_distances(Xr, center, cov)
    d_before = robust_mahalanobis_distances(Xb, center, cov)
    d_after = robust_mahalanobis_distances(Xa, center, cov)
    thr = np.quantile(d_real, tail_quantile)
    
    plt.figure(figsize=(8, 4))
    plt.hist(d_real, bins=50, alpha=0.5, label='real')
    plt.hist(d_before, bins=50, alpha=0.5, label='syn_before')
    plt.hist(d_after, bins=50, alpha=0.5, label='syn_after')
    plt.axvline(thr, color='r', ls='--', label='tail threshold')
    plt.legend()
    plt.title("Mahalanobis distance distributions")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()


def plot_tail_featurewise_analysis(df_real, df_syn_before, df_syn_after, tail_quantile=0.95):
    """Plot feature-wise tail distributions before and after correction."""
    cols = df_real.columns.tolist()
    scaler_obj = RobustScaler()
    Xr = scaler_obj.fit_transform(df_real.values)
    Xb = scaler_obj.transform(df_syn_before.values)
    Xa = scaler_obj.transform(df_syn_after.values)
    
    n_features = len(cols)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))
    
    plt.figure(figsize=(6 * n_cols, 3 * n_rows))
    for i, col in enumerate(cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        thr = np.quantile(Xr[:, i], tail_quantile)
        
        tail_real = Xr[Xr[:, i] >= thr, i]
        tail_before = Xb[Xb[:, i] >= thr, i]
        tail_after = Xa[Xa[:, i] >= thr, i]
        
        if len(tail_real) == 0 or len(tail_before) == 0 or len(tail_after) == 0:
            ax.text(0.5, 0.5, "Insufficient data\nin tail", ha='center', va='center')
            ax.set_title(f"{col} (empty tail)")
            ax.axis("off")
            continue
        
        bins = np.linspace(
            min(tail_real.min(), tail_before.min(), tail_after.min()),
            max(tail_real.max(), tail_before.max(), tail_after.max()),
            30
        )
        
        ax.hist(tail_real, bins=bins, alpha=0.5, label="real")
        ax.hist(tail_before, bins=bins, alpha=0.5, label="syn_before")
        ax.hist(tail_after, bins=bins, alpha=0.5, label="syn_after")
        ax.set_title(f"Tail ({col})")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
