"""
Utilities for dataset preprocessing and lightweight dataset/feature characterization.

This module provides `DatasetProcessor`, used by the AutoML pipeline to:
- clean the raw tabular data (drop NaNs, drop duplicates, label-encode categoricals)
- extract simple dataset-level and feature-level statistics used downstream
"""

import pandas as pd
from sklearn.metrics import mutual_info_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from toon_format import encode
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_integer_dtype,
    is_object_dtype,
)


def _to_python_scalar(x):
    """
    Convert common pandas/numpy scalar types into plain Python types so the output
    can be serialized reliably (JSON / pydantic / custom formats).
    """
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, np.generic):
        return x.item()

    return x


class DatasetProcessor:
    def __init__(self):
        """Create a dataset processor.

        The class is intentionally lightweight/stateless; methods operate on the
        provided `pandas.DataFrame` and return derived artifacts.
        """
        pass

    def _compute_mean_mutual_information(self, df: pd.DataFrame):
        """Compute mean pairwise mutual information across all columns.

        Notes:
            - Uses `sklearn.metrics.mutual_info_score`, which treats columns as
              discrete variables. Continuous/numerical columns are used as-is,
              which may not be statistically ideal but is sufficient as a simple
              dependency proxy.

        Args:
            df: Input dataframe.

        Returns:
            Mean mutual information over all unique column pairs, or 0 if there
            are fewer than 2 columns.
        """
        n = len(df.columns)
        if n < 2:
            return 0
        pairwise_mi = []
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if j > i:
                    mi = mutual_info_score(df[col1], df[col2])
                    pairwise_mi.append(mi)
        return np.mean(pairwise_mi) if pairwise_mi else 0

    def _extract_dataset_info(
        self,
        data: pd.DataFrame,
        feature_metadata: dict = None,
        target_column: str = None,
    ) -> dict:
        """Extract dataset-level and feature-level summary statistics.

        Dataset-level metrics include:
            - number of samples/features
            - categorical/numerical feature ratio (based on pandas dtypes)
            - mean/max absolute correlation between numerical features
            - mean pairwise mutual information between columns

        Feature-level metrics:
            - For categorical features (present in `feature_metadata`):
                - cardinality inferred from the stored label->class mapping
                - entropy computed from observed category probabilities
                - `rare_category_ratio`: probability mass of the rarest category
                - `unique_values`: explicit list of observed unique values for
                  low-cardinality columns
            - For numerical features:
                - mean/std/skewness/kurtosis/min/max/unique_ratio

        Args:
            data: Preprocessed dataframe (typically output of `_preprocess_data`).
            feature_metadata: Metadata produced by `_preprocess_data` for
                categorical columns (encoding type and label mapping). If None,
                all columns are treated as numerical for feature-level stats.
            target_column: Optional name of the target column; used only to tag
                feature roles as `"target"` vs `"feature"`.

        Returns:
            A nested dict with keys `"dataset_level"` and `"features_level"`.
        """
        processed_data = data.copy()
        max_unique_values_for_list = 50
        n_rows = len(processed_data)
        dataset_info = {
            "dataset_level": {},
            "features_level": {},
        }

        #### Dataset level ####

        # Number of samples
        n_samples = len(processed_data)
        n_features = len(processed_data.columns)

        def _is_categorical_column(col_name: str) -> bool:
            """Infer whether a column should be treated as categorical.

            Priority:
            1) explicit metadata (if provided)
            2) dtype-based checks (object/category/bool)
            3) heuristic for already encoded categorical ints
            """
            meta = feature_metadata.get(col_name) if feature_metadata else None
            if meta and meta.get("type") == "categorical":
                return True

            series = processed_data[col_name]

            if (
                is_object_dtype(series)
                or is_categorical_dtype(series)
                or is_bool_dtype(series)
            ):
                return True

            # Heuristic for encoded categorical integers when metadata is absent:
            # low cardinality compared to dataset size.
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

                # Generic low-cardinality heuristic for encoded categories.
                if (
                    nunique <= min(100, max(10, int(0.1 * n_rows)))
                    and unique_ratio <= 0.2
                ):
                    return True

                # Name-aware fallback: columns like state_code / class_id are often encoded categories.
                if has_categorical_name_hint and nunique <= 500 and unique_ratio <= 0.5:
                    return True

            return False

        cat_cols = [
            col for col in processed_data.columns if _is_categorical_column(col)
        ]
        num_cols = [col for col in processed_data.columns if col not in cat_cols]

        categorical_ratio = (
            round(len(cat_cols) / n_features, 4) if n_features > 0 else 0
        )
        numerical_ratio = round(len(num_cols) / n_features, 4) if n_features > 0 else 0

        # Mean and max absolute correlation between numerical features
        if n_features > 1:
            corr_matrix = processed_data.corr(numeric_only=True).abs()
            # Exclude diagonal (self-correlation)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            abs_corrs = corr_matrix.where(mask).stack()
            mean_abs_correlation = (
                round(abs_corrs.mean(), 4) if not abs_corrs.empty else 0
            )
            max_correlation = round(abs_corrs.max(), 4) if not abs_corrs.empty else 0
        else:
            mean_abs_correlation = 0
            max_correlation = 0

        mean_mutual_information = round(
            self._compute_mean_mutual_information(processed_data), 4
        )

        dataset_info["dataset_level"].update(
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "categorical_ratio": categorical_ratio,
                "numerical_ratio": numerical_ratio,
                "mean_abs_correlation": mean_abs_correlation,
                "max_correlation": max_correlation,
                "mean_mutual_information": mean_mutual_information,
            }
        )

        #### Features level ####

        for col in processed_data.columns:
            # Determine if the feature is a target or a feature using the target_column
            feature_role = "feature"
            if target_column:
                feature_role = "target" if col == target_column else "feature"

            # Get feature type, encoding and mapping
            meta = feature_metadata.get(col) if feature_metadata else None
            is_categorical = _is_categorical_column(col)
            if is_categorical:
                type = "categorical"
                encoding = meta.get("encoding") if meta else None
                mapping = meta.get("mapping", {}) if meta else {}
                cardinality = int(processed_data[col].nunique(dropna=True))

                # Compute entropy over observed category probabilities
                probs = (
                    processed_data[col]
                    .value_counts(normalize=True)
                    .to_numpy(dtype=float)
                )
                probs = probs[probs > 0]
                entropy = (
                    float(np.round(-np.sum(probs * np.log2(probs)), 4))
                    if probs.size
                    else 0.0
                )

                category_counts = processed_data[col].value_counts(normalize=True)
                rare_category_ratio = (
                    round(float(category_counts.min()), 4) if cardinality > 0 else 0
                )

                feature_info = {
                    "role": feature_role,
                    "type": type,
                    "encoding": encoding,
                    "mapping": mapping,
                    "cardinality": cardinality,
                    "entropy": entropy,
                    "rare_category_ratio": rare_category_ratio,
                }

                # Keep explicit value list for low-cardinality categories.
                # Also include common encoded categorical IDs (e.g., state_code)
                # even when they are slightly above the default threshold.
                col_name_lower = col.lower()
                has_categorical_name_hint = any(
                    token in col_name_lower
                    for token in ("code", "zone", "class", "category", "type", "label")
                )
                unique_values_limit = max(max_unique_values_for_list, 100)
                if cardinality <= unique_values_limit or (
                    has_categorical_name_hint and cardinality <= 500
                ):
                    unique_values = processed_data[col].dropna().unique().tolist()
                    unique_values = [_to_python_scalar(v) for v in unique_values]
                    feature_info["unique_values"] = sorted(
                        unique_values, key=lambda x: str(x)
                    )

                dataset_info["features_level"][col] = feature_info

            else:
                type = "numerical"

                # Compute feature statistics
                # Cast to plain Python scalars so serializers don't choke on numpy types.
                mean = _to_python_scalar(float(np.round(processed_data[col].mean(), 4)))
                std = _to_python_scalar(float(np.round(processed_data[col].std(), 4)))
                skewness = _to_python_scalar(
                    float(np.round(processed_data[col].skew(), 4))
                )
                kurtosis = _to_python_scalar(
                    float(np.round(processed_data[col].kurt(), 4))
                )
                min_value = _to_python_scalar(processed_data[col].min())
                max_value = _to_python_scalar(processed_data[col].max())
                unique_ratio = _to_python_scalar(
                    float(
                        np.round(processed_data[col].nunique() / len(processed_data), 4)
                    )
                )

                dataset_info["features_level"][col] = {
                    "role": feature_role,
                    "type": type,
                    "mean": mean,
                    "std": std,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "min": min_value,
                    "max": max_value,
                    "unique_ratio": unique_ratio,
                }

        return dataset_info

    def _preprocess_data(
        self, data: pd.DataFrame, verbose: bool = False
    ) -> tuple[pd.DataFrame, dict]:
        """
        Preprocess the input data by cleaning and validating it.

        Cleaning steps:
            - Drop rows containing any NaN values
            - Drop duplicate rows
            - Label-encode categorical columns (object/category/bool)

        Args:
            data: Input DataFrame to preprocess.
            verbose: If True, prints how many rows were removed.

        Returns:
            A tuple of:
                - processed_data: Cleaned dataframe.
                - feature_metadata: Dict describing categorical columns and their
                  encoding. For each encoded column includes:
                    - type: "categorical"
                    - encoding: "label"
                    - mapping: {encoded_label: original_class}

        Raises:
            ValueError: If all rows or all columns are removed after preprocessing.
        """
        feature_metadata = {}

        processed_data = data.copy()

        # Step 1: Remove any NaN values in samples
        initial_data_len = len(processed_data)
        processed_data = processed_data.dropna()
        removed_samples = initial_data_len - len(processed_data)

        if verbose and removed_samples > 0:
            print(f"Removed {removed_samples} samples with NaN values")

        # Step 2: Remove duplicate rows
        initial_rows = len(processed_data)
        processed_data = processed_data.drop_duplicates()
        removed_duplicates = initial_rows - len(processed_data)

        if verbose and removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")

        # Step 3: Encoding categorical columns if any
        categorical_columns = processed_data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                enc = LabelEncoder()
                processed_data[col] = enc.fit_transform(processed_data[col])
                feature_metadata[col] = {
                    "type": "categorical",
                    "encoding": "label",
                    "mapping": dict(zip(range(len(enc.classes_)), enc.classes_)),
                }

        # Validate that we still have data
        if len(processed_data) == 0:
            raise ValueError(
                "All data was removed during preprocessing. Check your dataset for quality issues."
            )

        if len(processed_data.columns) == 0:
            raise ValueError(
                "All features were removed during preprocessing. Check your dataset for quality issues."
            )

        return processed_data, feature_metadata

    def buld_models_dict(self, ctgan_syn_data, dpgan_syn_data, ddpm_syn_data) -> dict:
        ctgan_syn_data = ctgan_syn_data.to_dict(orient="records")
        dpgan_syn_data = dpgan_syn_data.to_dict(orient="records")
        ddpm_syn_data = ddpm_syn_data.to_dict(orient="records")

        ctgan_syn_data = encode(ctgan_syn_data)
        dpgan_syn_data = encode(dpgan_syn_data)
        ddpm_syn_data = encode(ddpm_syn_data)

        models_dict = {
            "ctgan": ctgan_syn_data,
            "dpgan": dpgan_syn_data,
            "ddpm": ddpm_syn_data,
        }

        return models_dict

    def build_dataset_info_dict(
        self, ctgan_syn_data, dpgan_syn_data, ddpm_syn_data
    ) -> dict:
        ctgan_syn_data_info = self._extract_dataset_info(ctgan_syn_data)
        dpgan_syn_data_info = self._extract_dataset_info(dpgan_syn_data)
        ddpm_syn_data_info = self._extract_dataset_info(ddpm_syn_data)

        dataset_info_dict = {
            "ctgan": ctgan_syn_data_info,
            "dpgan": dpgan_syn_data_info,
            "ddpm": ddpm_syn_data_info,
        }

        return dataset_info_dict
