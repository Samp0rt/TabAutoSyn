import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import wasserstein_distance, entropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
# ---------- Вспомогательные функции ----------
def js_divergence(p, q):
    """Дивергенция Йенсена–Шеннона между двумя распределениями."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def distribution_matrix(df, metric='mi', n_bins=10):
    """
    Считает матрицу попарных зависимостей (MI, Wasserstein, Jensen-Shannon)
    между признаками датафрейма.
    """
    cols = df.columns
    n = len(cols)
    mat = np.zeros((n, n))

    # Дискретизация для MI и JS
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_binned = est.fit_transform(df)

    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0.0 if metric == 'wasserstein' else 1.0
                continue

            x, y = df.iloc[:, i].values, df.iloc[:, j].values

            if metric == 'mi':
                mi = mutual_info_score(X_binned[:, i], X_binned[:, j])
                h_i = mutual_info_score(X_binned[:, i], X_binned[:, i])
                h_j = mutual_info_score(X_binned[:, j], X_binned[:, j])
                val = 0 if (h_i + h_j == 0) else 2 * mi / (h_i + h_j)

            elif metric == 'wasserstein':
                # Нормируем признак в [0, 1], чтобы значения были сопоставимы
                x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
                y_n = (y - y.min()) / (y.max() - y.min() + 1e-9)
                val = wasserstein_distance(x_n, y_n)

            elif metric == 'js':
                # Оценка плотности через гистограмму
                p, _ = np.histogram(x, bins=n_bins, range=(np.min(x), np.max(x)), density=True)
                q, _ = np.histogram(y, bins=n_bins, range=(np.min(y), np.max(y)), density=True)
                val = js_divergence(p + 1e-12, q + 1e-12)

            else:
                raise ValueError("Неизвестная метрика! Используй 'mi', 'wasserstein' или 'js'.")

            mat[i, j] = val

    # Нормализация: для Wasserstein — обратим шкалу, чтобы “похожее = 1, разное = 0”
    if metric == 'wasserstein':
        mat = 1 - mat / mat.max()

    return pd.DataFrame(mat, index=cols, columns=cols)


# ---------- Основная функция ----------
def matrix_eigenvalue_mse(real_df, synth_df, metric='mi', show_plots=False):
    """
    Сравнивает два датафрейма с использованием выбранной метрики:
    'mi' — взаимная информация,
    'wasserstein' — расстояние Вассерштейна,
    'js' — дивергенция Йенсена–Шеннона.
    """
    # Общие признаки и их порядок
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if not common_cols:
        raise ValueError("Нет общих признаков между real_df и synth_df!")
    real_df = real_df[common_cols]
    synth_df = synth_df[common_cols]

    # Матрицы метрик
    mat_real = distribution_matrix(real_df, metric=metric)
    mat_synth = distribution_matrix(synth_df, metric=metric)

    # Собственные значения
    eig_real = np.sort(np.real(np.linalg.eigvals(mat_real)))[::-1]
    eig_synth = np.sort(np.real(np.linalg.eigvals(mat_synth)))[::-1]
    mse = mean_squared_error(eig_real, eig_synth)

    # Визуализация
    if show_plots:
        plot_mi_and_eigs(mat_real, mat_synth, eig_real, eig_synth, mse, metric)

    return mat_real, mat_synth, eig_real, eig_synth, mse

def matrix_cosine_similarity(real_df, synth_df, metric='mi', show_plots=False):
    """
    Вычисляет косинусное сходство между матрицами реальных и синтетических данных.
    
    Parameters:
    -----------
    real_df : pd.DataFrame
        Датафрейм с реальными данными
    synth_df : pd.DataFrame
        Датафрейм с синтетическими данными
    metric : str, default='mi'
        Метрика для построения матриц ('mi', 'wasserstein', 'js')
    show_plots : bool, default=True
        Отображать ли визуализацию
    
    Returns:
    --------
    tuple : (mat_real, mat_synth, cosine_score, flattened_cosine)
        - mat_real: матрица метрик для реальных данных
        - mat_synth: матрица метрик для синтетических данных
        - cosine_score: среднее косинусное сходство построчно
        - flattened_cosine: косинусное сходство между векторизованными матрицами
    """
    # Общие признаки и их порядок
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if not common_cols:
        raise ValueError("Нет общих признаков между real_df и synth_df!")
    
    real_df = real_df[common_cols]
    synth_df = synth_df[common_cols]
    
    # Матрицы метрик (предполагается, что функция distribution_matrix определена)
    mat_real = distribution_matrix(real_df, metric=metric)
    mat_synth = distribution_matrix(synth_df, metric=metric)
    
    # Преобразование в numpy arrays если это DataFrame
    if isinstance(mat_real, pd.DataFrame):
        mat_real = mat_real.values
    if isinstance(mat_synth, pd.DataFrame):
        mat_synth = mat_synth.values
    
    # Векторизация матриц для вычисления косинусного сходства
    vec_real = mat_real.flatten().reshape(1, -1)
    vec_synth = mat_synth.flatten().reshape(1, -1)
    
    # Косинусное сходство между векторизованными матрицами
    flattened_cosine = cosine_similarity(vec_real, vec_synth)[0, 0]
    
    # Косинусное сходство построчно, затем среднее
    row_cosines = []
    for i in range(mat_real.shape[0]):
        cos_sim = cosine_similarity(
            mat_real[i].reshape(1, -1), 
            mat_synth[i].reshape(1, -1)
        )[0, 0]
        row_cosines.append(cos_sim)
    cosine_score = np.mean(row_cosines)
    
    # Визуализация
    if show_plots:
        plot_cosine_comparison(mat_real, mat_synth, cosine_score, 
                              flattened_cosine, metric)
    print(f"cosine_score: {cosine_score}")
    return mat_real, mat_synth, cosine_score, flattened_cosine


def matrix_frobenius_distance(real_df, synth_df, metric='mi', show_plots=False):
    """
    Вычисляет расстояние Фробениуса между матрицами реальных и синтетических данных.
    
    Parameters:
    -----------
    real_df : pd.DataFrame
        Датафрейм с реальными данными
    synth_df : pd.DataFrame
        Датафрейм с синтетическими данными
    metric : str, default='mi'
        Метрика для построения матриц ('mi', 'wasserstein', 'js')
    show_plots : bool, default=True
        Отображать ли визуализацию
    
    Returns:
    --------
    tuple : (mat_real, mat_synth, frobenius_dist, normalized_frobenius)
        - mat_real: матрица метрик для реальных данных
        - mat_synth: матрица метрик для синтетических данных
        - frobenius_dist: расстояние Фробениуса между матрицами
        - normalized_frobenius: нормализованное расстояние Фробениуса (0-1)
    """
    # Общие признаки и их порядок
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if not common_cols:
        raise ValueError("Нет общих признаков между real_df и synth_df!")
    
    real_df = real_df[common_cols]
    synth_df = synth_df[common_cols]
    
    mat_real = distribution_matrix(real_df, metric=metric)
    mat_synth = distribution_matrix(synth_df, metric=metric)
    
    # Преобразование в numpy arrays если это DataFrame
    if isinstance(mat_real, pd.DataFrame):
        mat_real = mat_real.values
    if isinstance(mat_synth, pd.DataFrame):
        mat_synth = mat_synth.values
    
    # Расстояние Фробениуса: ||A - B||_F = sqrt(sum((A - B)^2))
    diff_matrix = mat_real - mat_synth
    frobenius_dist = np.linalg.norm(diff_matrix, ord='fro')
    
    # Нормализованное расстояние Фробениуса (относительно нормы реальной матрицы)
    norm_real = np.linalg.norm(mat_real, ord='fro')
    normalized_frobenius = frobenius_dist / norm_real if norm_real > 0 else 0
    
    # Визуализация
    if show_plots:
        plot_frobenius_comparison(mat_real, mat_synth, frobenius_dist, 
                                 normalized_frobenius, metric)
    
    return mat_real, mat_synth, frobenius_dist, normalized_frobenius


# Вспомогательные функции визуализации (опциональные)
def plot_cosine_comparison(mat_real, mat_synth, cosine_score, 
                          flattened_cosine, metric):
    """Визуализация для косинусного сходства"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Реальная матрица
    im1 = axes[0].imshow(mat_real, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Реальные данные ({metric})')
    plt.colorbar(im1, ax=axes[0])
    
    # Синтетическая матрица
    im2 = axes[1].imshow(mat_synth, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Синтетические данные ({metric})')
    plt.colorbar(im2, ax=axes[1])
    
    # Разность матриц
    diff = mat_real - mat_synth
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('Разность матриц')
    plt.colorbar(im3, ax=axes[2])
    
    fig.suptitle(f'Косинусное сходство (построчное): {cosine_score:.4f}\n'
                f'Косинусное сходство (векторизованное): {flattened_cosine:.4f}',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_frobenius_comparison(mat_real, mat_synth, frobenius_dist, 
                              normalized_frobenius, metric):
    """Визуализация для расстояния Фробениуса"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Реальная матрица
    im1 = axes[0].imshow(mat_real, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Реальные данные ({metric})')
    plt.colorbar(im1, ax=axes[0])
    
    # Синтетическая матрица
    im2 = axes[1].imshow(mat_synth, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Синтетические данные ({metric})')
    plt.colorbar(im2, ax=axes[1])
    
    # Разность матриц
    diff = mat_real - mat_synth
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('Разность матриц')
    plt.colorbar(im3, ax=axes[2])
    
    fig.suptitle(f'Расстояние Фробениуса: {frobenius_dist:.4f}\n'
                f'Нормализованное расстояние: {normalized_frobenius:.4f}',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
