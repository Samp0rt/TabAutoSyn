import os
import joblib
import optuna
import pandas as pd
from typing import Optional
import logging

from .custom_metric import Metric  ### in progress ###
from .config import METRICS

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.distribution import LogDistribution
from synthcity.utils.optuna_sample import suggest_all
from synthcity.plugins import Plugins

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Optimize hyperparameters of tabular synthetic data generation plugins.

    This helper wraps Optuna to search the plugin hyperparameter space exposed
    by `synthcity.Plugins`. It evaluates candidate configurations using
    `synthcity.benchmark.Benchmarks` and user-provided metrics.
    """

    def __init__(
        self,
        output_folder: str = "./optimization_results",
        n_trials: int = 20,
        n_jobs: int = 1,
        log_params: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the hyperparameter optimizer.

        Args:
            output_folder: Base directory used for saving studies and artifacts
                when `log_params` is True.
            n_trials: Number of Optuna trials per plugin.
            n_jobs: Parallel workers for Optuna's `study.optimize`.
            log_params: If True, persist the resulting Optuna `Study` to
                `output_folder`.
            verbose: If True, emit informational logs during optimization.
        """
        self.output_folder = output_folder
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.log_params = log_params
        self.verbose = verbose

        # Create output directory if it doesn't exist
        if self.log_params:
            os.makedirs(self.output_folder, exist_ok=True)

        # Available plugins for optimization
        # self.available_plugins = Plugins().list()

        # GPU plugins (if CUDA is available)
        # self.gpu_plugins = ['ddpm', 'ctgan', 'dpgan'] if torch.cuda.is_available() else []

    def _save_study(self, df_name: str, plugin_name: str, study: optuna.Study) -> None:
        """
        Save an Optuna study to disk under a structured folder.

        Args:
            df_name (str): Name of the dataset
            plugin_name (str): Name of the plugin
            study (optuna.Study): The optimization study to save
        """
        save_folder = f"{self.output_folder}/studies/{df_name}/{plugin_name}/"
        os.makedirs(save_folder, exist_ok=True)
        joblib.dump(study, f"{save_folder}/{df_name}_{plugin_name}_study.pkl")

        if self.verbose:
            logger.info(f"Study saved to {save_folder}")

    def _create_objective_function(
        self, plugin_name: str, train_loader: GenericDataLoader, metrics: dict
    ) -> callable:
        """
        Create an objective function for Optuna optimization.

        Args:
            plugin_name: Name of the `synthcity` plugin to optimize.
            train_loader: Loader that wraps the training DataFrame and target.
            metrics: Mapping of metric names to `synthcity` metric objects
                consumed by `Benchmarks.evaluate`.

        Returns:
            A callable that Optuna will invoke per trial. It returns the
            aggregated score (lower is better) computed from metrics with
            direction "minimize" in the benchmark report.
        """

        def _objective(trial: optuna.Trial) -> float:
            """Objective function for hyperparameter optimization."""
            try:
                # Get hyperparameter space for the
                generators = Plugins()
                hp_space = generators.get(plugin_name).hyperparameter_space()

                # Special handling for DDPM
                if plugin_name == "ddpm":
                    hp_space[0] = LogDistribution(name="lr", low=1e-5, high=0.02)

                # Suggest hyperparameters
                params = suggest_all(trial, hp_space)

                if self.verbose:
                    print("-" * 60)
                    print(f'Trial {trial.number + 1} - Plugin "{plugin_name}"')
                    print(f"Train using parameters:\n{params}\n")

                trial_id = f"trial_{trial.number}"

                # Evaluate the plugin with suggested parameters
                report = Benchmarks.evaluate(
                    [(trial_id, plugin_name, params)],
                    train_loader,
                    repeats=3,
                    synthetic_cache=False,
                    metrics=metrics,
                )

                # Calculate score (average of minimize direction metrics)
                score = report[trial_id].query('direction == "minimize"')["mean"].mean()

                return score

            except Exception as e:
                if self.verbose:
                    logger.warning(
                        f"Trial {trial.number} failed: {type(e).__name__}: {e}"
                    )
                raise optuna.TrialPruned()

        return _objective

    def _optimize_plugin(
        self,
        plugin_name: str,
        train_data: pd.DataFrame,
        target_column: Optional[str] = None,
        df_name: str = "dataset",
    ) -> optuna.Study:
        """
        Run an Optuna search for a specific plugin on given training data.

        Args:
            plugin_name: Name of the plugin to optimize.
            train_data: Tabular training data as a pandas DataFrame.
            target_column: Optional target column name for supervised tasks.
            df_name: Dataset identifier used when persisting studies.

        Returns:
            The completed Optuna `Study` for this plugin.
        """
        metrics = METRICS

        # Create data loader
        train_loader = GenericDataLoader(train_data, target_column=target_column)

        # Create objective function
        objective_func = self._create_objective_function(
            plugin_name, train_loader, metrics
        )

        # Create and run optimization study
        study = optuna.create_study(direction="minimize")

        if self.verbose:
            logger.info(
                f"Starting optimization for plugin '{plugin_name}' with {self.n_trials} trials"
            )

        study.optimize(objective_func, n_trials=self.n_trials, n_jobs=self.n_jobs)

        # Save the study if log_params is True
        if self.log_params:
            self._save_study(df_name, plugin_name, study)

        if self.verbose:
            logger.info(f"Optimization completed for plugin '{plugin_name}'")
            logger.info(f"Best parameters: {study.best_params}")
            logger.info(f"Best score: {study.best_value}")

        return study
