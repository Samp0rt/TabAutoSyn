"""High-level TabAutoSyn API: tabular synthesis, genetic curation, and the async ``generate`` pipeline.

Integrates synthcity plugins, LLM-based dependency repair (:class:`~tabautosyn.agents.deps_reconstruction.DependencyFixer`),
tail extension, and optional Langfuse tracing for meta-model calls.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from tabautosyn.custom_metric import Metric  ### in progress ###
from tabautosyn.optimization import HyperparameterOptimizer
from tabautosyn.llm_generator import LLMGenerator
from tabautosyn.gen.gen import GAConfig, GeneticAlgorithm
from tabautosyn.tail_extension.tail import correct_tails_by_adding
from tabautosyn.utils.dataset_processor import DatasetProcessor
from tabautosyn.agents.prompts import (
    DEPENDENCY_DISCOVERY_PROMPT,
    USER_DF_INFO_GENERATOR_PROMPT,
)
from tabautosyn.agents.deps_reconstruction import DependencyFixer
from typing import Any

# Synthcity
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import synthcity.logger as sc_logger

sc_logger.remove()


def _suppress_synthcity_plugin_disabled_log() -> None:
    """Synthcity logs optional plugins as ``critical('module disabled: …')`` via loguru.

    Optional plugins (e.g. GOGGLE) are expected to be missing; the line is noise only.
    Other ``critical`` calls from :mod:`synthcity.logger` are forwarded unchanged.
    """
    _orig_critical = sc_logger.critical

    def _critical_filtered(*args: Any, **kwargs: Any) -> None:
        if args and isinstance(args[0], str) and args[0].startswith("module disabled:"):
            return
        return _orig_critical(*args, **kwargs)

    sc_logger.critical = _critical_filtered  # type: ignore[method-assign]


_suppress_synthcity_plugin_disabled_log()

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from rich.console import Console
from rich.rule import Rule
from rich.traceback import install as install_rich_traceback
from dotenv import load_dotenv

from tabautosyn.utils.langfuse import (
    get_langfuse_judge_client,
    langfuse_output_payload,
    langfuse_safe_end as _safe_langfuse_end,
    langfuse_safe_span as _safe_langfuse_span,
    langfuse_safe_trace as _safe_langfuse_trace,
    langfuse_safe_update as _safe_langfuse_update,
)

from openai import OpenAI
from rich import print

install_rich_traceback(show_locals=False)
load_dotenv()
RICH_CONSOLE = Console()


class TabAutoSyn:
    """Tabular synthesis orchestrator (synthcity plugins and/or LLM paths).

    Supports task-specific plugins (CTGAN, DDPM, DPGAN via ``model="task_specific"``) and
    LLM-based generation (``model="LLM"``). The primary end-to-end entry point for in-memory
    workflows is :meth:`generate`; file-based helpers include :meth:`run_generator` and
    :meth:`old_generate`.

    Attributes:
        model: ``"task_specific"`` or ``"LLM"``.
        task: For task-specific mode: ``"ml"``, ``"privacy"``, or ``"universal"``.
        verbose: If ``True``, print progress and Rich-enhanced tracebacks where applicable.
    """

    def __init__(
        self,
        model: str | None = None,
        task: str | None = None,
        verbose: bool | None = None,
    ):
        """Configure synthesis mode and logging verbosity.

        Args:
            model: ``"task_specific"`` (synthcity plugins) or ``"LLM"``. Defaults to ``"LLM"``.
            task: Used when ``model == "task_specific"``: ``"ml"``, ``"privacy"``, or ``"universal"``.
                Defaults to ``"universal"`` in that mode.
            verbose: Enable verbose logs. Defaults to ``False``.

        Raises:
            ValueError: If ``task`` is not one of the allowed values for task-specific mode,
                or if ``model`` is not a supported identifier.
        """
        model = "LLM" if model is None else model
        verbose = False if verbose is None else verbose

        # Valid tasks
        valid_tasks = ["ml", "privacy", "universal"]
        if model == "task_specific":
            task = "universal" if task is None else task
        if model == "task_specific" and task not in valid_tasks:
            raise ValueError(f"Invalid task '{task}'. Must be one of: {valid_tasks}")

        # Valid models
        valid_models = ["task_specific", "LLM"]
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")

        self.model = model
        self.task = task
        self.verbose = verbose

    def _generate_synthetics_llm(
        self,
        train_data: pd.DataFrame,
        plugin_name: str = "gpt-oss:20b",
        n_samples: int = 100,
        batch_size: int = 10,
        target_column: str = None,
    ) -> pd.DataFrame:
        """Generate synthetic rows via :class:`~tabautosyn.llm_generator.LLMGenerator` (local OpenAI-compatible API).

        The client is currently fixed to ``http://localhost:11434/v1`` (Ollama-style); ``plugin_name`` selects the model id.

        Args:
            train_data: Real data used as context for generation.
            plugin_name: Model identifier passed to the generator (e.g. ``"gpt-oss:20b"``).
            n_samples: Minimum number of rows to accumulate (loops until enough non-duplicate rows exist).
            batch_size: Rows requested per ``generate`` call inside the loop.
            target_column: If set, passed through to ``LLMGenerator`` for label-aware sampling.

        Returns:
            Concatenated synthetic dataframe (duplicates dropped when encountered).
        """
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        columns = train_data.columns

        if self.verbose:
            print(f'Start generating data using model "{plugin_name}"')

        generated_data = None
        while generated_data is None or len(generated_data) < n_samples:
            generator = LLMGenerator(
                gen_client=client,
                gen_model_nm=plugin_name,
                real_data=train_data,
                cols=columns,
                verbose=self.verbose,
                target_column=target_column,
            )

            if generated_data is None:
                generated_data = generator.generate(
                    n_samples=n_samples, batch_size=batch_size
                )
                if generated_data.duplicated().any():
                    generated_data = generated_data.drop_duplicates()
            else:
                additonal_samples = generator.generate(n_samples - len(generated_data))
                if additonal_samples.duplicated().any():
                    additonal_samples = additonal_samples.drop_duplicates()
                generated_data = pd.concat(
                    [generated_data, additonal_samples], axis=0, ignore_index=True
                )

        return generated_data

    def _plugin_task_type(self) -> str:
        """Synthcity DDPM flag: treat ``ml`` as classification, else regression."""
        return "classification" if self.task == "ml" else "regression"

    def _generate_synthetics_non_llm(
        self,
        train_data: pd.DataFrame,
        plugin_name: str,
        task_type: str | None = None,
        optimization_trials: int = None,
        target_column: str = None,
        n_samples: int = 100,
        custom_metric: Metric = None,
        params: str = None,
        log_params: bool = False,
        log_plugin_params: bool = True,
    ) -> pd.DataFrame:
        """Fit a synthcity plugin and sample synthetic rows (optional Optuna HPO).

        Args:
            train_data: Training data (typically already aligned with the plugin).
            plugin_name: Synthcity plugin name, e.g. ``"ctgan"``, ``"ddpm"``, ``"dpgan"``.
            task_type: ``"classification"`` or ``"regression"`` for DDPM; defaults from :meth:`_plugin_task_type`.
            optimization_trials: If set with ``params is None``, run HPO via :class:`~tabautosyn.optimization.HyperparameterOptimizer`.
            target_column: Target column name when training supervised plugins.
            n_samples: Minimum number of synthetic rows to return (may loop generation until reached).
            custom_metric: Reserved for custom quality objectives during HPO.
            params: Path to a pickled Optuna study (``joblib``) with ``best_params``; skips fresh HPO when set.
            log_params: When ``True``, persist optimization artifacts according to the optimizer configuration.
            log_plugin_params: When ``True`` and ``verbose``, print plugin hyperparameters before fitting.

        Returns:
            Synthetic samples as a DataFrame with a reset index.

        Raises:
            ValueError: If loading ``params`` from disk fails.
        """

        init_kwargs = {
            "ctgan": {},
            "ddpm": {},
            "dpgan": {},
        }

        # Step 1: Hyperparameter optimization
        if optimization_trials != None and params == None:
            try:
                # Create optimizer
                optimizer = HyperparameterOptimizer(
                    output_folder="./optimization_results",
                    n_jobs=1,
                    log_params=log_params,
                    n_trials=optimization_trials,
                    verbose=self.verbose,
                )

                # Run optimization
                optimization_result = optimizer._optimize_plugin(
                    plugin_name=plugin_name,
                    train_data=train_data,
                    target_column=target_column,
                    df_name="optimized_dataset",
                )

                init_kwargs[plugin_name] = optimization_result.best_params

                if self.verbose:
                    print("Hyperparameter optimization completed!")
                    print(
                        f"Best parameters for {plugin_name}: {optimization_result.best_params}"
                    )
                    print(
                        f"Best score for {plugin_name}: {optimization_result.best_value}"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Hyperparameter optimization failed: {str(e)}")

        # Step 2: Extracting existing params
        if params != None:
            try:
                optimization_result = joblib.load(params)
                init_kwargs[plugin_name] = optimization_result.best_params
            except:
                raise ValueError(
                    f"Error while using predefined Optuna params: {str(e)}"
                )

        task_type = task_type if task_type is not None else self._plugin_task_type()

        # Step 3: Training plugin (ctgan, ddpm or dpgan)
        if (
            self.verbose
            and log_plugin_params
            and (optimization_trials is not None or params is not None)
        ):
            print(
                f'Start training model "{plugin_name}". Using parameters:\n{init_kwargs[plugin_name]}'
            )

        generators = Plugins()

        train_loader = GenericDataLoader(train_data, target_column=target_column)

        if plugin_name == "ddpm" and task_type == "classification":
            init_kwargs["ddpm"]["is_classification"] = True

        # Check if optimization parameters were provided for this plugin
        has_optimization_params = (
            plugin_name in init_kwargs and init_kwargs[plugin_name]
        )

        def _synthcity_fit(gen, loader):
            """Synthcity uses tqdm/rich on stderr; keep streams open when *verbose*."""
            if self.verbose:
                gen.fit(loader)
            else:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    gen.fit(loader)

        def _synthcity_generate(gen, n):
            """Sample *n* rows from a fitted synthcity generator, optionally silencing tqdm/rich."""
            if self.verbose:
                return gen.generate(n)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return gen.generate(n)

        try:
            syn_df = None
            while syn_df is None or len(syn_df) < n_samples:
                # Use optimization parameters if available
                if has_optimization_params:
                    generator = generators.get(
                        plugin_name,
                        compress_dataset=False,
                        strict=False,
                        **init_kwargs[plugin_name],
                    )
                else:
                    generator = generators.get(
                        plugin_name,
                        compress_dataset=False,
                        strict=False,
                    )
                _synthcity_fit(generator, train_loader)

                if syn_df is None:
                    syn_df = _synthcity_generate(generator, n_samples)
                    syn_df = syn_df.dataframe()
                    syn_df = syn_df.dropna()
                    if syn_df.duplicated().any():
                        syn_df = syn_df.drop_duplicates()
                else:
                    additonal_samples = _synthcity_generate(
                        generator, n_samples - len(syn_df)
                    )
                    additonal_samples = additonal_samples.dataframe()
                    additonal_samples = additonal_samples.dropna()
                    if additonal_samples.duplicated().any():
                        additonal_samples = additonal_samples.drop_duplicates()

                    syn_df = pd.concat(
                        [syn_df, additonal_samples], axis=0, ignore_index=True
                    )

        except ValueError as e:
            if has_optimization_params:
                print("\nTrying fitting generator without optimization parameters...")
                try:
                    syn_df = None
                    while syn_df is None or len(syn_df) < n_samples:
                        generator = generators.get(
                            plugin_name,
                            compress_dataset=False,
                            strict=False,
                        )
                        _synthcity_fit(generator, train_loader)

                        if syn_df is None:
                            syn_df = _synthcity_generate(generator, n_samples)
                            syn_df = syn_df.dataframe()
                            syn_df = syn_df.dropna()
                            if syn_df.duplicated().any():
                                syn_df = syn_df.drop_duplicates()
                        else:
                            additonal_samples = _synthcity_generate(
                                generator, n_samples - len(syn_df)
                            )
                            additonal_samples = additonal_samples.dataframe()
                            additonal_samples = additonal_samples.dropna()
                            if additonal_samples.duplicated().any():
                                additonal_samples = additonal_samples.drop_duplicates()
                            syn_df = pd.concat(
                                [syn_df, additonal_samples], axis=0, ignore_index=True
                            )

                except Exception as e:
                    print(
                        f"Synthetic data generation failed: {str(e)}. Please try another model"
                    )
            else:
                print(
                    f"Synthetic data generation failed: {str(e)}. Please try another model"
                )

        return syn_df.reset_index(drop=True)

    def _perform_curation(
        self,
        syn_data: pd.DataFrame,
        real_data: pd.DataFrame,
        n_generations: int = 20,
        crossover_prob: float | int = 1,
        bootstrap_sample_ratio: list = None,  # int | float = 0.9,
        n_bootstrap_samples: int = 10,
        target_column: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Run genetic curation to align synthetic rows with real data (see :class:`~tabautosyn.gen.gen.GeneticAlgorithm`).

        Args:
            syn_data: Synthetic candidate dataframe.
            real_data: Real reference dataframe for fitness comparison.
            n_generations: GA generations.
            crossover_prob: Crossover probability passed to :class:`~tabautosyn.gen.gen.GAConfig`.
            bootstrap_sample_ratio: Bootstrap sizing controls for GA (see ``GAConfig``).
            n_bootstrap_samples: Number of bootstrap draws per fitness evaluation context.
            target_column: Supervised label column used by the GA when applicable.
            verbose: Forwarded to ``GAConfig`` / genetic algorithm logging.

        Returns:
            Curated synthetic dataframe produced by ``GeneticAlgorithm.run``.
        """

        config = GAConfig(
            n_generations=n_generations,
            crossover_prob=crossover_prob,
            bootstrap_sample_ratio=bootstrap_sample_ratio,
            n_bootstrap_samples=n_bootstrap_samples,
            verbose=verbose,
        )
        ga = GeneticAlgorithm(config=config, target_col=target_column)
        results = ga.run(syn_data, real_data)

        return results

    def _extract_outliers(self, df: pd.DataFrame, columns: Any = None, threshold=1.5):
        """Return rows that are outliers on at least one numeric column (IQR rule).

        Args:
            df: Input dataframe.
            columns: Columns to scan; if ``None``, all numeric columns are used.
            threshold: IQR multiplier for lower/upper fences (default ``1.5``).

        Returns:
            Subset of *df* flagged as outliers on any selected column, with NA rows dropped.
        """
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Initialize mask with all False
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            # Calculate Q1, Q3, and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier boundaries
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Identify outliers for this column
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers

        # Extract outlier rows
        outliers_df = df[outlier_mask].copy()

        outliers_df = outliers_df.dropna()

        return outliers_df

    def old_generate(
        self,
        train_data_path: str = None,
        sep: str = ",",
        n_samples: int | None = None,
        batch_size: int = 10,
        log_params: bool = False,
        custom_metric: Metric = None,
        optimization_trials: int = None,
        params: str = None,
        target_column: str = None,
        n_generations: int = 20,
        crossover_prob: float | int = 1,
        bootstrap_sample_ratio: list = None,  # int | float = 0.9,
    ):
        """Legacy CSV-based pipeline: load data, synthesize, tails, class filter, genetic curation.

        Prefer :meth:`generate` for the async LLM + dependency-repair pipeline, or :meth:`run_generator`
        for controlled file I/O without that stack.

        Args:
            train_data_path: Path to a CSV training file.
            sep: CSV delimiter.
            n_samples: Target synthetic row count (non-LLM path may oversample then trim).
            batch_size: LLM batch size when ``model == "LLM"``.
            log_params: Persist HPO studies when optimizing plugin hyperparameters.
            custom_metric: Reserved custom metric hook.
            optimization_trials: Optuna trial count for plugin HPO.
            params: Pickled study path for warm-started plugin params.
            target_column: Label column for supervised checks and curation.
            n_generations: GA generations in :meth:`_perform_curation`.
            crossover_prob: GA crossover probability.
            bootstrap_sample_ratio: GA bootstrap ratio configuration.

        Returns:
            Final curated synthetic dataframe.

        Raises:
            ValueError: On missing path, missing ``n_samples``, or generation/preprocessing errors.

        Example:
            >>> syn = TabAutoSyn(model="task_specific", task="ml", verbose=False)
            >>> df = syn.old_generate("train.csv", n_samples=200, target_column="label")  # doctest: +SKIP
        """
        if train_data_path is not None:
            if n_samples is not None:
                try:
                    # Load data with error handling
                    train_data = pd.read_csv(train_data_path, sep=sep)

                    if self.verbose:
                        print(
                            f"\nLoaded dataset with {len(train_data)} samples and {len(train_data.columns)} features"
                        )

                    # Optimized preprocessing pipeline
                    train_data_mod = self._preprocess_data(train_data)

                    if self.verbose:
                        print(
                            f"After preprocessing: {len(train_data_mod)} samples and {len(train_data_mod.columns)} features"
                        )
                        print(
                            f"Removed {len(train_data) - len(train_data_mod)} samples during preprocessing"
                        )

                    # Perform hyperparameter optimization for non-LLM model
                    if optimization_trials != None and params == None:
                        if self.verbose and self.model != "LLM":
                            print("Starting hyperparameter optimization...")

                    if self.model == "task_specific":
                        if self.task == "privacy":
                            plugin_name = "dpgan"
                        elif self.task == "ml":
                            plugin_name = "ddpm"
                        elif self.task == "universal":
                            plugin_name = "ctgan"

                    outliers = self._extract_outliers(
                        df=train_data_mod, columns=train_data_mod.columns
                    )

                    if self.model == "task_specific":
                        syn_df = self._generate_synthetics_non_llm(
                            train_data=train_data_mod,
                            plugin_name=plugin_name,
                            task_type=self._plugin_task_type(),
                            optimization_trials=optimization_trials,
                            target_column=target_column,
                            n_samples=round(n_samples * 1.5),
                            custom_metric=custom_metric,
                            params=params,
                            log_params=log_params,
                            log_plugin_params=True,
                        )

                        syn_outliers = self._generate_synthetics_non_llm(
                            train_data=outliers,
                            plugin_name=plugin_name,
                            task_type=self._plugin_task_type(),
                            optimization_trials=optimization_trials,
                            target_column=target_column,
                            n_samples=len(outliers),
                            custom_metric=custom_metric,
                            params=params,
                            log_params=log_params,
                            log_plugin_params=False,
                        )

                    elif self.model == "LLM":
                        syn_df = self._generate_synthetics_llm(
                            train_data=train_data_mod,
                            n_samples=len(train_data_mod),
                            batch_size=batch_size,
                            target_column=target_column,
                        )

                        syn_outliers = self._generate_synthetics_llm(
                            train_data=outliers,
                            n_samples=len(outliers),
                            batch_size=batch_size,
                            target_column=target_column,
                        )

                    # check NaNs and duplicates
                    if syn_df.isnull().any().any():
                        syn_df = syn_df.dropna()
                    if syn_outliers.isnull().any().any():
                        syn_outliers = syn_outliers.dropna()
                    if syn_df.duplicated().any():
                        syn_df = syn_df.drop_duplicates()
                    if syn_outliers.duplicated().any():
                        syn_outliers = syn_outliers.drop_duplicates()

                    syn_df_with_tails, _, _ = correct_tails_by_adding(
                        df_real=train_data_mod,
                        df_syn=syn_df,
                        df_syn_tail=syn_outliers,
                        divergence_metric="js",
                        loss_scope="hybrid",
                        verbose=self.verbose,
                    )

                    # Check classes for classification
                    real_classes = set(train_data_mod[target_column].unique())
                    syn_classes = set(syn_df_with_tails[target_column].unique())

                    missing_in_syn = real_classes - syn_classes
                    extra_in_syn = syn_classes - real_classes

                    if not missing_in_syn and not extra_in_syn:
                        if self.verbose:
                            print("\nAll classes are similar in both datasets.")
                        synthetic_filtered = syn_df_with_tails.copy()
                        train_data_filtered = train_data_mod.copy()
                    else:
                        if self.verbose:
                            if missing_in_syn:
                                print(
                                    f"\nThere are missing classes in synthetic dataset: {missing_in_syn}"
                                )
                            if extra_in_syn:
                                print(
                                    f"\nThere are extra classes in synthetic dataset: {extra_in_syn}"
                                )

                        common_classes = real_classes.intersection(syn_classes)
                        synthetic_filtered = syn_df_with_tails[
                            syn_df_with_tails[target_column].isin(common_classes)
                        ].copy()
                        train_data_filtered = train_data_mod[
                            train_data_mod[target_column].isin(common_classes)
                        ].copy()

                        if self.verbose:
                            print(
                                f"Real data target classes: {np.sort(train_data_filtered[target_column].unique())}"
                            )
                            print(
                                f"Synthetic dataset target classes: {np.sort(synthetic_filtered[target_column].unique())}"
                            )

                    # Start curation process
                    if self.verbose:
                        print(f"\nStarting evolutional optimization ...")

                    syn_df_final = self._perform_curation(
                        syn_data=synthetic_filtered,
                        real_data=train_data_filtered,
                        target_column=target_column,
                        n_generations=n_generations,
                        crossover_prob=crossover_prob,
                        bootstrap_sample_ratio=bootstrap_sample_ratio,
                        verbose=self.verbose,
                    )

                    return syn_df_final

                except Exception as e:
                    raise ValueError(f"Error generating synthetic data: {str(e)}")

            else:
                raise ValueError("n_samples must be provided")

        else:
            raise ValueError("train_data_path is required")

    def run_generator(
        self,
        train_data_path: str = None,
        run_preprocessing: bool = True,
        generate_tails: bool = False,
        sep: str = ",",
        n_samples: int | None = None,
        batch_size: int = 10,
        log_params: bool = False,
        custom_metric: Metric = None,
        optimization_trials: int = None,
        params: str = None,
        target_column: str = None,
    ):
        """Load a CSV and return either full synthetic data or a synthetic outlier pool.

        When ``generate_tails`` is ``True``, fits generators on IQR outliers only and returns that
        dataframe; otherwise fits on (optionally preprocessed) training data and returns the main synthetic set.

        Args:
            train_data_path: Path to input CSV.
            run_preprocessing: If ``True``, run :class:`~tabautosyn.utils.dataset_processor.DatasetProcessor` preprocessing.
            generate_tails: If ``True``, synthesize from :meth:`_extract_outliers` rows instead of the full frame.
            sep: CSV delimiter.
            n_samples: Row count target (required when ``generate_tails`` is ``False``).
            batch_size: LLM batch size for ``model == "LLM"``.
            log_params: Persist HPO studies for plugin optimization.
            custom_metric: Reserved metric hook for HPO.
            optimization_trials: Optuna trials when optimizing plugin hyperparameters.
            params: Path to pickled best params for warm start.
            target_column: Supervised target for plugin / LLM generation.

        Returns:
            Synthetic dataframe for the chosen branch.

        Raises:
            ValueError: On missing ``n_samples``, read/processing errors, or invalid configuration.
        """

        if train_data_path is not None:
            try:
                # Load data with error handling
                train_data = pd.read_csv(train_data_path, sep=sep)

                if self.verbose:
                    print(
                        f"\nLoaded dataset with {len(train_data)} samples and {len(train_data.columns)} features"
                    )

                if run_preprocessing:
                    dataset_processor = DatasetProcessor()
                    train_data_mod, _ = dataset_processor._preprocess_data(
                        train_data, verbose=self.verbose
                    )

                    if self.verbose:
                        print(
                            f"After preprocessing: {len(train_data_mod)} samples and {len(train_data_mod.columns)} features"
                        )
                        print(
                            f"Removed {len(train_data) - len(train_data_mod)} samples during preprocessing"
                        )

                # Perform hyperparameter optimization for non-LLM model
                if optimization_trials != None and params == None:
                    if self.verbose and self.model != "gpt-oss":
                        print("Starting hyperparameter optimization...")

                if self.model == "task_specific":
                    if self.task == "privacy":
                        plugin_name = "dpgan"
                    elif self.task == "ml":
                        plugin_name = "ddpm"
                    elif self.task == "universal":
                        plugin_name = "ctgan"

                if generate_tails:
                    outliers = self._extract_outliers(
                        df=train_data_mod if run_preprocessing else train_data,
                        columns=(
                            train_data_mod.columns
                            if run_preprocessing
                            else train_data.columns
                        ),
                    )

                    outliers = outliers.dropna()

                    if self.model == "task_specific":
                        syn_outliers = self._generate_synthetics_non_llm(
                            train_data=outliers,
                            plugin_name=plugin_name,
                            task_type=self._plugin_task_type(),
                            optimization_trials=optimization_trials,
                            target_column=target_column,
                            n_samples=(
                                len(outliers)
                                if len(outliers) < n_samples
                                else n_samples
                            ),
                            custom_metric=custom_metric,
                            params=params,
                            log_params=log_params,
                            log_plugin_params=True,
                        )

                        return syn_outliers

                    elif self.model == "LLM":
                        syn_outliers = self._generate_synthetics_llm(
                            train_data=outliers,
                            n_samples=(
                                len(outliers)
                                if len(outliers) < n_samples
                                else n_samples
                            ),
                            batch_size=batch_size,
                            target_column=target_column,
                        )

                        return syn_outliers

                else:
                    if n_samples is not None:
                        if self.model == "task_specific":
                            syn_df = self._generate_synthetics_non_llm(
                                train_data=(
                                    train_data_mod if run_preprocessing else train_data
                                ),
                                plugin_name=plugin_name,
                                task_type=self._plugin_task_type(),
                                optimization_trials=optimization_trials,
                                target_column=target_column,
                                n_samples=round(n_samples * 1.5),
                                custom_metric=custom_metric,
                                params=params,
                                log_params=log_params,
                                log_plugin_params=True,
                            )

                            return syn_df

                        elif self.model == "LLM":
                            syn_df = self._generate_synthetics_llm(
                                train_data=(
                                    train_data_mod if run_preprocessing else train_data
                                ),
                                n_samples=round(n_samples * 1.5),
                                batch_size=batch_size,
                                target_column=target_column,
                            )

                            return syn_df

                    else:
                        raise ValueError("n_samples must be provided")

            except Exception as e:
                raise ValueError(f"Error while generating synthetic data: {str(e)}")

    def run_outliers_extension(
        self,
        real_data: pd.DataFrame | None = None,
        syn_data_full: pd.DataFrame | None = None,
        syn_outliers: pd.DataFrame | None = None,
        divergence_metric: str = "js",
        compare_metric: str = "mi",
        loss_scope: str = "hybrid",
    ):
        """Blend synthetic tail/outlier rows into the main synthetic body using :func:`~tabautosyn.tail_extension.tail.correct_tails_by_adding`.

        Args:
            real_data: Reference real dataframe.
            syn_data_full: Main synthetic sample pool.
            syn_outliers: Synthetic rows aligned with real outliers (tails).
            divergence_metric: Divergence used inside tail correction (e.g. ``"js"``).
            compare_metric: Comparison metric forwarded to ``correct_tails_by_adding``.
            loss_scope: Hybrid / tail loss scope for the tail extension routine.

        Returns:
            Dataframe with tails merged into the synthetic distribution.

        Raises:
            ValueError: If any of the three dataframes is ``None``, or if tail correction fails.
        """
        if real_data is None or syn_data_full is None or syn_outliers is None:
            raise ValueError(
                "Missing required data: real_data, syn_data_full, and syn_outliers must all be provided"
            )

        try:
            syn_df_with_tails, _, _ = correct_tails_by_adding(
                df_real=real_data,
                df_syn=syn_data_full,
                df_syn_tail=syn_outliers,
                compare_metric=compare_metric,
                divergence_metric=divergence_metric,
                loss_scope=loss_scope,
                verbose=self.verbose,
            )
            return syn_df_with_tails

        except Exception as e:
            raise ValueError(f"Error while extending synthetic outliers: {str(e)}")

    def run_evolutional_optimization(
        self,
        real_data: pd.DataFrame | None = None,
        syn_data: pd.DataFrame | None = None,
        n_generations: int = 20,
        crossover_prob: float | int = 1,
        bootstrap_sample_ratio: list = None,  # int | float = 0.9,
        target_column: str = None,
    ):
        """Align target classes between real and synthetic data, then run genetic curation.

        Filters both frames to the intersection of label values when mismatches exist, then calls
        :meth:`_perform_curation` with the configured GA hyperparameters.

        Args:
            real_data: Reference dataframe (must include ``target_column``).
            syn_data: Synthetic dataframe (must include ``target_column``).
            n_generations: Number of GA generations.
            crossover_prob: GA crossover probability.
            bootstrap_sample_ratio: GA bootstrap configuration (see ``GAConfig``).
            target_column: Name of the classification target column.

        Returns:
            Curated synthetic dataframe after GA.

        Note:
            *real_data* and *syn_data* must both contain ``target_column``.
        """
        real_classes = set(real_data[target_column].unique())
        syn_classes = set(syn_data[target_column].unique())

        missing_in_syn = real_classes - syn_classes
        extra_in_syn = syn_classes - real_classes

        if not missing_in_syn and not extra_in_syn:
            if self.verbose:
                print("\nAll classes are similar in both datasets.")
            synthetic_filtered = syn_data.copy()
            real_data_filtered = real_data.copy()
        else:
            if self.verbose:
                if missing_in_syn:
                    print(
                        f"\nThere are missing classes in synthetic dataset: {missing_in_syn}"
                    )
                if extra_in_syn:
                    print(
                        f"\nThere are extra classes in synthetic dataset: {extra_in_syn}"
                    )

            common_classes = real_classes.intersection(syn_classes)
            synthetic_filtered = syn_data[
                syn_data[target_column].isin(common_classes)
            ].copy()
            real_data_filtered = real_data[
                real_data[target_column].isin(common_classes)
            ].copy()

            if self.verbose:
                print(
                    f"Real data target classes: {np.sort(real_data_filtered[target_column].unique())}"
                )
                print(
                    f"Synthetic dataset target classes: {np.sort(synthetic_filtered[target_column].unique())}"
                )

        # Start curation process
        if self.verbose:
            print(f"\nStarting evolutional optimization ...")

        syn_df_final = self._perform_curation(
            syn_data=synthetic_filtered,
            real_data=real_data_filtered,
            n_generations=n_generations,
            crossover_prob=crossover_prob,
            bootstrap_sample_ratio=bootstrap_sample_ratio,
            target_column=target_column,
            verbose=self.verbose,
        )

        return syn_df_final

    async def generate(
        self,
        train_data: pd.DataFrame = None,
        user_df_info: str = None,
        target_column: str = None,
        log_params: bool = False,
        custom_metric: Metric = None,
        optimization_trials: int = None,
        params: str = None,
        temperature: float = 0.1,
        max_tokens: int = 16000,
        retries: int = 3,
        timeout: int = 120,
        save_pipeline_summary: bool = False,
        ouput_dir: str | None = None,
    ):
        """End-to-end async pipeline: synthcity generation, LLM dependency repair, tails, curation, optional Langfuse.

        Expects in-memory ``train_data`` (no CSV I/O here). Runs configured plugins (see in-method ``plugins`` list),
        discovers and fixes structural dependencies with Pydantic-AI agents, extends tails, matches classification
        labels to the real distribution, runs genetic curation, and optionally writes a Markdown summary and CSV.

        Args:
            train_data: Real training dataframe (required).
            user_df_info: Short natural-language dataset description; if omitted, may be generated via LLM.
            target_column: Required supervised column for class-consistency filtering and plugin training.
            log_params: Passed through to non-LLM generation for HPO logging.
            custom_metric: Reserved for plugin optimization.
            optimization_trials: Optuna trials for plugin HPO when applicable.
            params: Pickled study path for warm-started synthcity params.
            temperature: OpenRouter chat sampling temperature for meta-agents.
            max_tokens: Max tokens for meta-agent completions.
            retries: Pydantic-AI agent retry count for dependency discovery.
            timeout: Per-request timeout (seconds) for OpenRouter-backed models.
            save_pipeline_summary: If ``True``, write a run summary next to outputs when ``ouput_dir`` is set.
            ouput_dir: Output directory for summary CSV/MD (typo preserved for backward compatibility).

        Returns:
            Final synthetic dataframe (reset index).

        Raises:
            RuntimeError: If ``OPENROUTER_API_KEY`` is missing.
            ValueError: If ``train_data`` is ``None``, ``target_column`` is invalid, or the pipeline fails.

        Note:
            Langfuse tracing uses :func:`~tabautosyn.utils.langfuse.get_langfuse_judge_client`; initialization
            failures are non-fatal and only disable tracing.
        """
        if train_data is not None:
            langfuse_client = None
            pipeline_trace = None
            try:
                pipeline_started_at = datetime.now()
                plugins = ["ctgan", "ddpm", "dpgan"]
                plugin_summaries: list[dict[str, Any]] = []
                try:
                    langfuse_client = get_langfuse_judge_client()
                    pipeline_trace = _safe_langfuse_trace(
                        langfuse_client,
                        name="tabautosyn.generate",
                        input_payload={
                            "n_input_rows": len(train_data),
                            "n_input_columns": len(train_data.columns),
                            "target_column": target_column,
                            "plugins": plugins,
                        },
                        metadata={"component": "TabAutoSyn.generate"},
                    )
                except Exception:
                    if self.verbose:
                        print(
                            "[yellow]Langfuse tracing initialization skipped due to error.[/yellow]"
                        )

                api_key = os.getenv("OPENROUTER_API_KEY") or ""
                default_meta_model = os.getenv(
                    "DEFAULT_META_MODEL", "anthropic/claude-haiku-4.5"
                )

                if not api_key:
                    raise RuntimeError(
                        "No OPENROUTER_API_KEY provided. Set it in environment or pass as argument."
                    )

                dependency_discovery_model = _create_model(
                    model=default_meta_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key,
                )

                encoding_checker_model = _create_model(
                    model=default_meta_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key,
                )

                dependency_violation_detector_model = _create_model(
                    model=default_meta_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key,
                )

                user_df_generator_model = _create_model(
                    model=default_meta_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key,
                )

                if self.verbose:
                    print(
                        f"\nLoaded dataset with {len(train_data)} samples and {len(train_data.columns)} features"
                    )

                dataset_processor = DatasetProcessor()
                train_data_mod, _ = dataset_processor._preprocess_data(train_data, verbose=self.verbose)

                if self.verbose:
                    print(
                        f"After preprocessing: {len(train_data_mod)} samples and {len(train_data_mod.columns)} features"
                    )
                    print(
                        f"Removed {len(train_data) - len(train_data_mod)} samples during preprocessing"
                    )

                train_data = train_data_mod

                if target_column is None:
                    raise ValueError(
                        "target_column is required for class consistency checks and curation."
                    )

                if target_column:
                    if not target_column in train_data.columns:
                        raise ValueError(
                            f"target_column {target_column} not found in train_data columns. Please provide a valid target_column."
                        )

                outliers = self._extract_outliers(
                    df=train_data, columns=train_data.columns
                )
                if self.verbose:
                    print(
                        f"[cyan]Outlier extraction complete:[/cyan] {len(outliers)} rows "
                        f"from {len(train_data)} preprocessed rows."
                    )

                final_syn_df = pd.DataFrame()

                for plugin_idx, plugin_name in enumerate(plugins, start=1):
                    plugin_span = _safe_langfuse_span(
                        pipeline_trace,
                        name=f"plugin.{plugin_name}",
                        input_payload={"plugin": plugin_name},
                    )
                    if self.verbose:
                        print(
                            f"\n[bold cyan]🔌 Plugin[/bold cyan] [yellow]{plugin_idx}/{len(plugins)}[/yellow] "
                            f"[bold white]·[/bold white] [green]{plugin_name}[/green]"
                        )
                    syn_df = pd.DataFrame()
                    syn_outliers = pd.DataFrame()

                    syn_df = self._generate_synthetics_non_llm(
                        train_data=train_data,
                        plugin_name=plugin_name,
                        task_type=self._plugin_task_type(),
                        optimization_trials=optimization_trials,
                        target_column=target_column,
                        n_samples=len(train_data),
                        custom_metric=custom_metric,
                        params=params,
                        log_params=log_params,
                        log_plugin_params=True,
                    )
                    if self.verbose:
                        print(
                            f"[green]{plugin_name} synthetic generation done:[/green] "
                            f"{len(syn_df)} rows."
                        )
                    _safe_langfuse_update(
                        plugin_span,
                        metadata={"generated_rows": len(syn_df)},
                    )

                    syn_outliers = self._generate_synthetics_non_llm(
                        train_data=outliers,
                        plugin_name=plugin_name,
                        task_type=self._plugin_task_type(),
                        optimization_trials=optimization_trials,
                        target_column=target_column,
                        n_samples=len(outliers),
                        custom_metric=custom_metric,
                        params=params,
                        log_params=log_params,
                        log_plugin_params=False,
                    )
                    if self.verbose:
                        print(
                            f"[green]{plugin_name} outlier synthesis done:[/green] "
                            f"{len(syn_outliers)} rows."
                        )
                    _safe_langfuse_update(
                        plugin_span,
                        metadata={"generated_outlier_rows": len(syn_outliers)},
                    )

                    # check NaNs and duplicates
                    syn_rows_before = len(syn_df)
                    outlier_rows_before = len(syn_outliers)
                    if syn_df.isnull().any().any():
                        syn_df = syn_df.dropna()
                    if syn_outliers.isnull().any().any():
                        syn_outliers = syn_outliers.dropna()
                    if syn_df.duplicated().any():
                        syn_df = syn_df.drop_duplicates()
                    if syn_outliers.duplicated().any():
                        syn_outliers = syn_outliers.drop_duplicates()
                    if self.verbose:
                        print(
                            f"[cyan]Cleanup ({plugin_name}):[/cyan] main {syn_rows_before} -> {len(syn_df)}, "
                            f"outliers {outlier_rows_before} -> {len(syn_outliers)}."
                        )

                    real_data_info_dict = dataset_processor._extract_dataset_info(
                        train_data, target_column=target_column
                    )
                    str_real_data_info_dict = json.dumps(
                        real_data_info_dict, ensure_ascii=False
                    )
                    syn_data_cols = str(syn_df.columns.tolist())

                    if not user_df_info:
                        if self.verbose:
                            print(
                                "[cyan]Generating user_df_info from real data chunk...[/cyan]"
                            )
                        real_data_chunk = train_data.sample(
                            n=min(10, len(train_data)), random_state=42
                        )
                        str_real_data_chunk = json.dumps(
                            real_data_chunk.to_dict(orient="records"),
                            ensure_ascii=False,
                        )
                        user_df_info_generator = Agent(
                            name="UserDfInfoGenerator",
                            model=user_df_generator_model,
                            system_prompt=(
                                USER_DF_INFO_GENERATOR_PROMPT.safe_substitute(
                                    columns=syn_data_cols,
                                    real_data_chunk=str_real_data_chunk,
                                )
                            ),
                            instrument=True,
                        )
                        user_df_info_result = await user_df_info_generator.run(
                            "Generate dataset description."
                        )
                        user_df_info = (
                            str(user_df_info_result.output)
                            .strip()
                            .splitlines()[0]
                            .strip()
                            .strip('"')
                        )
                        if self.verbose:
                            print(
                                f"[green]Generated user_df_info:[/green] {user_df_info}"
                            )
                    elif self.verbose:
                        print("[cyan]Using provided user_df_info.[/cyan]")

                    if self.verbose:
                        RICH_CONSOLE.print()
                        RICH_CONSOLE.print(
                            Rule(
                                "[bold magenta]🔗 Dependency discovery[/bold magenta] "
                                f"[dim]· plugin {plugin_name} ·[/dim]",
                                style="magenta",
                            )
                        )

                    dep_discovery_span = _safe_langfuse_trace(
                        langfuse_client,
                        name="DependencyDiscoveryAgent",
                        input_payload={
                            "plugin": plugin_name,
                            "columns": (
                                syn_data_cols[:4000]
                                if len(syn_data_cols) > 4000
                                else syn_data_cols
                            ),
                        },
                        metadata={
                            "plugin": plugin_name,
                            "source": "tabautosyn.generate",
                            "agent": "DependencyDiscoveryAgent",
                        },
                        new_trace=True,
                    )
                    try:
                        dependency_discovery_agent = Agent(
                            name="DependencyDiscoveryAgent",
                            model=dependency_discovery_model,
                            system_prompt=(
                                DEPENDENCY_DISCOVERY_PROMPT.safe_substitute(
                                    domain_description=user_df_info,
                                    columns=syn_data_cols,
                                    statistics=str_real_data_info_dict,
                                )
                            ),
                            retries=retries,
                            instrument=False,
                        )

                        dependency_discovery_result = (
                            await dependency_discovery_agent.run()
                        )
                        dependency_discovery_result = str(
                            dependency_discovery_result.output
                        ).strip()
                        if dependency_discovery_result.startswith("```json"):
                            dependency_discovery_result = dependency_discovery_result[
                                len("```json") :
                            ].strip()
                        elif dependency_discovery_result.startswith("```"):
                            dependency_discovery_result = dependency_discovery_result[
                                len("```") :
                            ].strip()
                        if dependency_discovery_result.endswith("```"):
                            dependency_discovery_result = dependency_discovery_result[
                                :-3
                            ].strip()
                        dependency_discovery_result = json.loads(
                            dependency_discovery_result
                        )

                        dep_summary = {
                            dep_type: len(dep_items)
                            for dep_type, dep_items in dependency_discovery_result.get(
                                "dependencies", {}
                            ).items()
                            if dep_items
                        }
                        _deps_out = langfuse_output_payload(
                            dependency_discovery_result.get("dependencies") or {},
                            key="dependencies",
                        )
                        _safe_langfuse_update(
                            dep_discovery_span,
                            output_payload={
                                "dependency_summary": dep_summary,
                                "dependency_types": list(
                                    (
                                        dependency_discovery_result.get("dependencies")
                                        or {}
                                    ).keys()
                                ),
                                **_deps_out,
                            },
                        )
                    except Exception as e:
                        _safe_langfuse_update(
                            dep_discovery_span,
                            output_payload={"error": str(e)},
                            level="ERROR",
                            status_message=str(e),
                        )
                        raise
                    finally:
                        _safe_langfuse_end(dep_discovery_span)
                    if self.verbose:
                        RICH_CONSOLE.print(
                            f"[magenta]Dependency discovery summary[/magenta] "
                            f"[dim]· {plugin_name} ·[/dim] {dep_summary}"
                        )
                        RICH_CONSOLE.print()
                        RICH_CONSOLE.print(
                            Rule(
                                "[bold yellow]🛠 Dependency fixer[/bold yellow] "
                                f"[dim]· plugin {plugin_name} ·[/dim]",
                                style="yellow",
                            )
                        )

                    deps_fixer = DependencyFixer(
                        syn_df=syn_df,
                        real_df=train_data,
                        dependencies=dependency_discovery_result["dependencies"],
                    )
                    fixed_syn_df = await deps_fixer.fix_dependencies_async(
                        user_df_info=user_df_info,
                        encoding_checker_model=encoding_checker_model,
                        dependency_violation_detector_model=dependency_violation_detector_model,
                        real_df=train_data,
                        verbose=self.verbose,
                        segment_label="Main synthetic (full_df)",
                        langfuse_client=langfuse_client,
                        langfuse_encoding_metadata={"plugin": plugin_name},
                    )

                    outlier_deps_fixer = DependencyFixer(
                        syn_df=syn_outliers,
                        real_df=train_data,
                        dependencies=dependency_discovery_result["dependencies"],
                    )
                    fixed_syn_outliers = await outlier_deps_fixer.fix_dependencies_async(
                        user_df_info=user_df_info,
                        encoding_checker_model=encoding_checker_model,
                        dependency_violation_detector_model=dependency_violation_detector_model,
                        real_df=train_data,
                        verbose=self.verbose,
                        segment_label="Tail outliers",
                        langfuse_client=langfuse_client,
                        langfuse_encoding_metadata={"plugin": plugin_name},
                    )
                    if self.verbose:
                        RICH_CONSOLE.print(
                            f"[cyan]Dependency fixer — row counts ({plugin_name})[/cyan]\n"
                            f"  [bold]Main synthetic (full_df):[/bold] [green]{len(fixed_syn_df)}[/green] rows\n"
                            f"  [bold]Tail outliers:[/bold]           [green]{len(fixed_syn_outliers)}[/green] rows"
                        )

                    syn_df_with_tails, _, _ = correct_tails_by_adding(
                        df_real=train_data,
                        df_syn=fixed_syn_df,
                        df_syn_tail=fixed_syn_outliers,
                        divergence_metric="js",
                        loss_scope="hybrid",
                        verbose=self.verbose,
                    )

                    # Check classes for classification
                    real_classes = set(train_data[target_column].unique())
                    syn_classes = set(syn_df_with_tails[target_column].unique())

                    missing_in_syn = real_classes - syn_classes
                    extra_in_syn = syn_classes - real_classes

                    if not missing_in_syn and not extra_in_syn:
                        if self.verbose:
                            print("\nAll classes are similar in both datasets.")
                        synthetic_filtered = syn_df_with_tails.copy()
                        train_data_filtered = train_data.copy()
                    else:
                        if self.verbose:
                            if missing_in_syn:
                                print(
                                    f"\nThere are missing classes in synthetic dataset: {missing_in_syn}"
                                )
                            if extra_in_syn:
                                print(
                                    f"\nThere are extra classes in synthetic dataset: {extra_in_syn}"
                                )

                        common_classes = real_classes.intersection(syn_classes)
                        synthetic_filtered = syn_df_with_tails[
                            syn_df_with_tails[target_column].isin(common_classes)
                        ].copy()
                        train_data_filtered = train_data[
                            train_data[target_column].isin(common_classes)
                        ].copy()

                        if self.verbose:
                            print(
                                f"Real data target classes: {np.sort(train_data_filtered[target_column].unique())}"
                            )
                            print(
                                f"Synthetic dataset target classes: {np.sort(synthetic_filtered[target_column].unique())}"
                            )
                    if self.verbose:
                        print(
                            f"[cyan]Post-class filtering ({plugin_name}):[/cyan] "
                            f"real={len(train_data_filtered)}, synthetic={len(synthetic_filtered)}."
                        )

                    # Start curation process
                    if self.verbose:
                        RICH_CONSOLE.print()
                        RICH_CONSOLE.print(
                            Rule(
                                "[bold green]🧬 Genetic curation[/bold green] "
                                f"[dim]· plugin {plugin_name} ·[/dim]",
                                style="green",
                            )
                        )
                        RICH_CONSOLE.print(
                            "[dim]Evolutionary optimization vs real data (population fitness)…[/dim]"
                        )

                    syn_df_curated = self._perform_curation(
                        syn_data=synthetic_filtered,
                        real_data=train_data_filtered,
                        target_column=target_column,
                        verbose=self.verbose,
                    )
                    if self.verbose:
                        print(
                            f"[green]Curation complete ({plugin_name}):[/green] "
                            f"{len(syn_df_curated)} rows."
                        )

                    final_syn_df = pd.concat([final_syn_df, syn_df_curated])
                    if self.verbose:
                        print(
                            f"[bold green]Accumulated synthetic rows:[/bold green] {len(final_syn_df)}"
                        )
                    plugin_summaries.append(
                        {
                            "plugin": plugin_name,
                            "generated_rows": len(syn_df),
                            "generated_outlier_rows": len(syn_outliers),
                            "fixed_rows": len(fixed_syn_df),
                            "fixed_outlier_rows": len(fixed_syn_outliers),
                            "post_tail_rows": len(syn_df_with_tails),
                            "post_filter_real_rows": len(train_data_filtered),
                            "post_filter_syn_rows": len(synthetic_filtered),
                            "curated_rows": len(syn_df_curated),
                            "dependency_summary": dep_summary,
                        }
                    )
                    _safe_langfuse_update(
                        plugin_span,
                        output_payload={
                            "curated_rows": len(syn_df_curated),
                            "post_filter_syn_rows": len(synthetic_filtered),
                            "dependency_summary": dep_summary,
                        },
                    )
                    _safe_langfuse_end(plugin_span)

                pipeline_finished_at = datetime.now()
                should_save_artifacts = save_pipeline_summary or bool(ouput_dir)
                summary_path = None
                dataset_path = None

                if should_save_artifacts:
                    artifact_dir = ouput_dir or os.path.join(
                        "tabautosyn_logs", "pipeline_summaries"
                    )
                    os.makedirs(artifact_dir, exist_ok=True)
                    summary_path = os.path.join(
                        artifact_dir,
                        f"generate_summary_{pipeline_finished_at.strftime('%Y%m%d_%H%M%S')}.md",
                    )
                    dataset_path = os.path.join(
                        artifact_dir,
                        f"final_synthetic_{pipeline_finished_at.strftime('%Y%m%d_%H%M%S')}.csv",
                    )

                    summary_md = _markdown_generate_pipeline_summary(
                        pipeline_started_at=pipeline_started_at,
                        pipeline_finished_at=pipeline_finished_at,
                        n_input_rows=len(train_data),
                        n_input_columns=len(train_data.columns),
                        target_column=target_column,
                        n_outliers=len(outliers),
                        plugins=plugins,
                        final_rows=len(final_syn_df),
                        dataset_path=dataset_path,
                        plugin_summaries=plugin_summaries,
                    )

                    with open(summary_path, "w", encoding="utf-8") as summary_file:
                        summary_file.write(summary_md)
                    final_syn_df.reset_index(drop=True).to_csv(
                        dataset_path, index=False
                    )

                if self.verbose:
                    print(
                        f"[bold green]Generation pipeline finished.[/bold green] "
                        f"Final rows: {len(final_syn_df)}"
                    )
                    if summary_path:
                        print(
                            f"[bold blue]Pipeline summary saved:[/bold blue] {summary_path}"
                        )
                    if dataset_path:
                        print(
                            f"[bold blue]Final synthetic dataset saved:[/bold blue] {dataset_path}"
                        )
                _safe_langfuse_update(
                    pipeline_trace,
                    output_payload={
                        "final_rows": len(final_syn_df),
                        "summary_path": summary_path,
                        "dataset_path": dataset_path,
                        "plugins": plugin_summaries,
                        "duration_seconds": (
                            pipeline_finished_at - pipeline_started_at
                        ).total_seconds(),
                    },
                    metadata={"status": "success"},
                )
                _safe_langfuse_end(pipeline_trace)
                if langfuse_client and hasattr(langfuse_client, "flush"):
                    langfuse_client.flush()
                return final_syn_df.reset_index(drop=True)

            except Exception as e:
                RICH_CONSOLE.print_exception(show_locals=False)
                _safe_langfuse_update(
                    pipeline_trace,
                    metadata={"status": "error", "error_type": type(e).__name__},
                    output_payload={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
                _safe_langfuse_end(pipeline_trace)
                if langfuse_client and hasattr(langfuse_client, "flush"):
                    langfuse_client.flush()
                raise ValueError(f"Error generating synthetic data: {str(e)}")

        else:
            raise ValueError("train_data is required")


def _md_table_row(cells: list[str]) -> str:
    """One markdown table row; escapes ``|`` in cell text."""
    safe = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
    return "| " + " | ".join(safe) + " |"


def _markdown_generate_pipeline_summary(
    *,
    pipeline_started_at: datetime,
    pipeline_finished_at: datetime,
    n_input_rows: int,
    n_input_columns: int,
    target_column: str | None,
    n_outliers: int,
    plugins: list[str],
    final_rows: int,
    dataset_path: str | None,
    plugin_summaries: list[dict[str, Any]],
) -> str:
    """Build a readable Markdown report for :meth:`TabAutoSyn.generate` artifact export."""
    duration_s = (pipeline_finished_at - pipeline_started_at).total_seconds()
    lines: list[str] = [
        "# TabAutoSyn — `generate` pipeline summary",
        "",
        f"> Run **{pipeline_started_at.isoformat(timespec='seconds')}** → **{pipeline_finished_at.isoformat(timespec='seconds')}** "
        f"· **{duration_s:.2f} s** · **{final_rows:,}** synthetic rows",
        "",
        "---",
        "",
        "## Run overview",
        "",
        _md_table_row(["Metric", "Value"]),
        _md_table_row(["---", "---:"]),
        _md_table_row(["Input rows", f"{n_input_rows:,}"]),
        _md_table_row(["Input columns", str(n_input_columns)]),
        _md_table_row(
            ["Target column", f"`{target_column}`" if target_column else "—"]
        ),
        _md_table_row(["Outlier pool (IQR)", f"{n_outliers:,}"]),
        _md_table_row(["Plugins", ", ".join(f"`{p}`" for p in plugins)]),
        _md_table_row(["Final CSV", f"`{dataset_path}`" if dataset_path else "—"]),
        "",
        "## Plugins",
        "",
    ]

    if not plugin_summaries:
        lines.append("*No plugin summaries were recorded.*")
        lines.append("")

    stage_labels = [
        ("generated_rows", "Generated (main)"),
        ("generated_outlier_rows", "Generated (outliers)"),
        ("fixed_rows", "After dependency fixer (main)"),
        ("fixed_outlier_rows", "After dependency fixer (outliers)"),
        ("post_tail_rows", "After tail correction"),
        ("post_filter_real_rows", "After class filter — real"),
        ("post_filter_syn_rows", "After class filter — synthetic"),
        ("curated_rows", "After genetic curation"),
    ]

    for idx, item in enumerate(plugin_summaries, start=1):
        pname = str(item.get("plugin", f"plugin_{idx}"))
        lines.append(f"### {idx}. `{pname}`")
        lines.append("")
        lines.append(_md_table_row(["Stage", "Rows"]))
        lines.append(_md_table_row(["---", "---:"]))
        for key, label in stage_labels:
            val = item.get(key, "—")
            lines.append(
                _md_table_row([label, f"{val:,}" if isinstance(val, int) else str(val)])
            )
        lines.append("")
        deps = item.get("dependency_summary") or {}
        lines.append("#### Discovered dependencies")
        lines.append("")
        if deps:
            lines.append(_md_table_row(["Type", "Count"]))
            lines.append(_md_table_row(["---", "---:"]))
            for dep_type, dep_count in sorted(deps.items(), key=lambda x: str(x[0])):
                lines.append(
                    _md_table_row(
                        [
                            f"`{dep_type}`",
                            (
                                f"{dep_count:,}"
                                if isinstance(dep_count, int)
                                else str(dep_count)
                            ),
                        ]
                    )
                )
        else:
            lines.append("*No non-empty dependency groups in this run.*")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("*Generated by TabAutoSyn `generate`.*")
    return "\n".join(lines)


def _create_model(model, temperature, max_tokens, timeout, api_key) -> OpenAIChatModel:
    """Build an OpenRouter-backed :class:`~pydantic_ai.models.openai.OpenAIChatModel` for Pydantic-AI agents."""
    settings = ModelSettings(
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    return OpenAIChatModel(
        model,
        provider=OpenRouterProvider(api_key=api_key),
        settings=settings,
    )
