"""TabNAT synthetic data generation for the TabAutoSyn pipeline."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_integer_dtype,
    is_object_dtype,
)
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from tabautosyn.generators.tabnat.model import TabNAT

TABNAT_DEFAULTS: dict[str, Any] = dict(
    embed_dim=32,
    buffer_size=8,
    depth=6,
    dropout_rate=0.0,
    lr=1e-3,
    weight_decay=1e-6,
    batch_size=1024,
    epochs=5000,
    seed=42,
)


def _choose_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _infer_cat_cols(df: pd.DataFrame, target_column: str | None) -> list[str]:
    n_rows = len(df)

    def _is_categorical_column(col_name: str) -> bool:
        series = df[col_name]
        if (
            is_object_dtype(series)
            or is_categorical_dtype(series)
            or is_bool_dtype(series)
        ):
            return True
        if is_integer_dtype(series):
            nunique = int(series.nunique(dropna=True))
            if n_rows == 0:
                return False
            unique_ratio = nunique / n_rows
            col_name_lower = col_name.lower()
            has_categorical_name_hint = any(
                token in col_name_lower
                for token in ("code", "zone", "class", "category", "type", "label")
            )
            if (
                nunique <= min(100, max(10, int(0.1 * n_rows)))
                and unique_ratio <= 0.2
            ):
                return True
            if has_categorical_name_hint and nunique <= 500 and unique_ratio <= 0.5:
                return True
        return False

    return [
        col
        for col in df.columns
        if col != target_column and _is_categorical_column(col)
    ]


def _map_task_type(
    df: pd.DataFrame, target_column: str, pipeline_task: str
) -> str:
    if pipeline_task == "regression":
        return "regression"
    y = df[target_column]
    if int(y.nunique()) == 2:
        return "binary_classification"
    return "multiclass_classification"


class _TabularDataset(Dataset):
    def __init__(self, x_num, x_cat):
        self.x_num = x_num
        self.x_cat = x_cat

    def __getitem__(self, index: int):
        if self.x_num is None:
            return self.x_cat[index]
        if self.x_cat is None:
            return self.x_num[index]
        return self.x_num[index], self.x_cat[index]

    def __len__(self) -> int:
        if self.x_num is not None:
            return self.x_num.shape[0]
        return self.x_cat.shape[0]


def _preprocess_tabnat(
    df: pd.DataFrame,
    target: str,
    cat_cols: list[str],
    task_type: str,
    seed: int,
):
    feature_cols = [c for c in df.columns if c != target]
    cat_features = [c for c in cat_cols if c in feature_cols]
    num_features = [c for c in feature_cols if c not in cat_features]
    is_reg = task_type == "regression"

    if is_reg:
        num_full = (
            pd.concat([df[[target]], df[num_features]], axis=1)
            if num_features or True
            else None
        )
        cat_full = df[cat_features] if cat_features else None
    else:
        num_full = df[num_features] if num_features else None
        cat_full = (
            pd.concat([df[[target]], df[cat_features]], axis=1)
            if cat_features
            else df[[target]]
        )

    if num_full is not None and num_full.shape[1] > 0:
        num_arr = num_full.to_numpy(dtype=np.float64)
        n_q = max(2, min(1000, num_arr.shape[0]))
        qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            random_state=seed,
            subsample=int(1e9),
        )
        num_q = qt.fit_transform(num_arr).astype(np.float32)
        mean = num_q.mean(0)
        std = num_q.std(0)
        std = np.where(std < 1e-8, 1.0, std)
        num_norm = (num_q - mean) / std / 2.0
        x_num_t = torch.tensor(num_norm).float()

        def num_inverse(arr: np.ndarray, _mean=mean, _std=std, _qt=qt) -> np.ndarray:
            arr = arr * 2.0 * _std + _mean
            return _qt.inverse_transform(arr)

        num_columns = list(num_full.columns)
    else:
        x_num_t = None
        num_inverse = None
        num_columns = []

    if cat_full is not None and cat_full.shape[1] > 0:
        cat_encoders: list[tuple[str, LabelEncoder]] = []
        cat_codes = np.zeros((cat_full.shape[0], cat_full.shape[1]), dtype=np.int64)
        for j, col in enumerate(cat_full.columns):
            le = LabelEncoder()
            vals = cat_full[col].astype(str).to_numpy()
            cat_codes[:, j] = le.fit_transform(vals)
            cat_encoders.append((col, le))
        x_cat_t = torch.tensor(cat_codes).long()
        categories = [int(le.classes_.shape[0]) for _, le in cat_encoders]
        cat_columns = list(cat_full.columns)

        def cat_inverse(arr: np.ndarray, _enc=cat_encoders) -> np.ndarray:
            out = np.empty(arr.shape, dtype=object)
            for j, (_, le) in enumerate(_enc):
                codes = np.clip(arr[:, j].astype(int), 0, len(le.classes_) - 1)
                out[:, j] = le.inverse_transform(codes)
            return out
    else:
        x_cat_t = None
        cat_inverse = None
        categories = []
        cat_columns = []

    meta = dict(
        num_columns=num_columns,
        cat_columns=cat_columns,
        target=target,
        task_type=task_type,
        feature_order=list(df.columns),
        num_inverse=num_inverse,
        cat_inverse=cat_inverse,
        categories=categories,
    )
    return x_num_t, x_cat_t, meta


def _train_tabnat_model(
    model,
    x_num,
    x_cat,
    device: str,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    verbose: bool,
    log_every: int = 200,
):
    dataset = _TabularDataset(x_num, x_cat)
    batch_size = min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(optim, mode="min", factor=0.9, patience=50)

    has_num = x_num is not None
    has_cat = x_cat is not None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        for batch in loader:
            if has_num and has_cat:
                batch_num, batch_cat = batch
            elif has_num:
                batch_num, batch_cat = batch, None
            else:
                batch_num, batch_cat = None, batch
            if batch_num is not None:
                batch_num = batch_num.to(device)
            if batch_cat is not None:
                batch_cat = batch_cat.to(device)
            optim.zero_grad()
            loss, _, _ = model(batch_num, batch_cat)
            loss.backward()
            optim.step()
            n = (batch_num if batch_num is not None else batch_cat).shape[0]
            epoch_loss += loss.item() * n
            epoch_n += n
        epoch_loss /= max(epoch_n, 1)
        sched.step(epoch_loss)
        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            lr_cur = optim.param_groups[0]["lr"]
            print(
                f"    tabnat ep {epoch + 1:5d}/{epochs}  loss={epoch_loss:.4f}  lr={lr_cur:.2e}",
                flush=True,
            )
    return model


def _sample_tabnat(
    model, n_synth: int, device: str, *, chunk: int = 2048
) -> tuple[np.ndarray | None, np.ndarray | None]:
    num_chunks: list[torch.Tensor] = []
    cat_chunks: list[torch.Tensor] = []
    done = 0
    while done < n_synth:
        cur = min(chunk, n_synth - done)
        syn_num, syn_cat = model.sample(cur, cls=None, device=device)
        if syn_num is not None:
            num_chunks.append(syn_num.cpu())
        if syn_cat is not None:
            cat_chunks.append(syn_cat.cpu())
        done += cur
    num_np = torch.cat(num_chunks, 0).numpy() if num_chunks else None
    cat_np = torch.cat(cat_chunks, 0).numpy() if cat_chunks else None
    return num_np, cat_np


def _reconstruct_tabnat_df(num_np, cat_np, meta) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    if num_np is not None and meta["num_inverse"] is not None:
        num_vals = meta["num_inverse"](num_np)
        for j, col in enumerate(meta["num_columns"]):
            data[col] = num_vals[:, j]
    if cat_np is not None and meta["cat_inverse"] is not None:
        cat_vals = meta["cat_inverse"](cat_np)
        for j, col in enumerate(meta["cat_columns"]):
            data[col] = cat_vals[:, j]
    return pd.DataFrame(data)[meta["feature_order"]]


def generate_tabnat_synthetics(
    train_data: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
    n_samples: int,
    verbose: bool = False,
    seed: int | None = None,
    device: str | None = None,
    cat_cols: list[str] | None = None,
    tabnat_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train TabNAT on *train_data* and return *n_samples* synthetic rows."""
    cfg = {**TABNAT_DEFAULTS, **(tabnat_kwargs or {})}
    seed = int(seed if seed is not None else cfg["seed"])
    device = _choose_device(device or os.getenv("TABNAT_DEVICE"))
    cat_cols = cat_cols if cat_cols is not None else _infer_cat_cols(
        train_data, target_column
    )
    tabnat_task = _map_task_type(train_data, target_column, task_type)

    df = train_data.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("TabNAT training data is empty after dropna().")

    syn_df = None
    while syn_df is None or len(syn_df) < n_samples:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_num, x_cat, meta = _preprocess_tabnat(
            df, target_column, cat_cols, tabnat_task, seed=seed
        )
        n_num = x_num.shape[1] if x_num is not None else 0
        n_cat = x_cat.shape[1] if x_cat is not None else 0
        if n_num == 0 and n_cat == 0:
            raise ValueError("TabNAT found no numeric or categorical features.")

        def _build_and_run():
            model = TabNAT(
                n_num=n_num,
                n_cat=n_cat,
                categories=meta["categories"],
                embed_dim=cfg["embed_dim"],
                buffer_size=cfg["buffer_size"],
                depth=cfg["depth"],
                norm_layer=nn.LayerNorm,
                dropout_rate=cfg["dropout_rate"],
                device=device,
            ).to(device)
            _train_tabnat_model(
                model,
                x_num,
                x_cat,
                device,
                batch_size=cfg["batch_size"],
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                verbose=verbose,
            )
            model.eval()
            with torch.no_grad():
                need = n_samples if syn_df is None else n_samples - len(syn_df)
                num_np, cat_np = _sample_tabnat(model, need, device)
            return _reconstruct_tabnat_df(num_np, cat_np, meta)

        if verbose:
            batch = _build_and_run()
        else:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                batch = _build_and_run()

        batch = batch.dropna()
        if batch.duplicated().any():
            batch = batch.drop_duplicates()

        if syn_df is None:
            syn_df = batch
        else:
            syn_df = pd.concat([syn_df, batch], axis=0, ignore_index=True)

        if device == "cuda":
            torch.cuda.empty_cache()

    return syn_df.iloc[:n_samples].reset_index(drop=True)
