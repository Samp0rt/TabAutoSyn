import pandas as pd
import numpy as np
import joblib
from tabautosyn.custom_metric import Metric  ### in progress ###
from tabautosyn.optimization import HyperparameterOptimizer
from tabautosyn.outliers import ExtractOutliers  ### in progress ###
from tabautosyn.llm_generator import LLMGenerator
from tabautosyn.gen.gen import GAConfig, GeneticAlgorithm
from tabautosyn.tail_extension.tail import correct_tails_by_adding

from typing import Any

# Synthcity
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from tabautosyn.models.ctgan import CTGAN

from openai import OpenAI


class TabAutoSyn:
    """
    TabAutoSyn: A tabular data synthesis framework for automated machine learning.

    This class provides functionality for generating synthetic tabular data using
    various models. It supports different tasks including
    machine learning, privacy preservation, and universal data generation.

    Attributes:
        model (str): The synthesis model to use. Options: "task_specific", "LLM"
        task (str): The type of task for data synthesis. Options: "ml", "privacy", "universal"
        verbose (bool): Whether to print detailed progress information during processing

    Example:
        >>> synthesizer = TabAutoSyn(model="task_specific", task="ml", verbose=True)
        >>> synthetic_data = synthesizer.generate("data.csv", n_samples=1000)
    """


    def __init__(
        self,
        model: str = "LLM",
        task: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the TabAutoSyn synthesizer.

        Args:
            model (str, optional): The synthesis model to use.
                "task_specific": Uses task-specific model
                "LLM": Uses large language model-based synthesis

                Defaults to "LLM"

            task (str, optional): The type of task for data synthesis if model "task-specific" is set
                "ml": Machine learning focused synthesis
                "privacy": Privacy-preserving data generation
                "universal": General-purpose data synthesis

                Defaults to "universal"

            verbose (bool, optional): Whether to print detailed progress information
                during data processing and synthesis.

                Defaults to False.

        Raises:
            ValueError: If the provided task is not in the list of valid tasks.
        """
        # Valid tasks
        valid_tasks = ["ml", "privacy", "universal"]
        if model == "task_specific" and task not in valid_tasks:
            raise ValueError(f"Invalid task '{task}'. Must be one of: {valid_tasks}")

        # Valid models
        valid_models = ["task_specific", "LLM"]
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")

        self.model = model
        self.task = task
        self.verbose = verbose


    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data by cleaning and validating it.

        This private method performs essential data cleaning steps including:
        - Removing completely empty columns (features)
        - Removing completely empty rows (samples)
        - Removing duplicate rows
        - Validating that the resulting dataset is not empty

        Args:
            data (pd.DataFrame): Input DataFrame to preprocess.

        Returns:
            pd.DataFrame: Cleaned and validated DataFrame.

        Raises:
            ValueError: If all data or all features are removed during preprocessing,
                indicating serious data quality issues.

        Note:
            Operates on a copy and leaves the original `data` unchanged.
        """
        processed_data = data.copy()

        # Step 1: Remove completely empty columns (features)
        initial_cols = len(processed_data.columns)
        processed_data = processed_data.dropna(axis=1, how="all")
        removed_cols = initial_cols - len(processed_data.columns)

        if self.verbose and removed_cols > 0:
            print(f"Removed {removed_cols} empty columns")

        # Step 2: Remove completely empty rows (samples)
        initial_rows = len(processed_data)
        processed_data = processed_data.dropna(axis=0, how="all")
        removed_rows = initial_rows - len(processed_data)

        if self.verbose and removed_rows > 0:
            print(f"Removed {removed_rows} empty rows")

        # Step 3: Remove duplicate rows
        initial_rows = len(processed_data)
        processed_data = processed_data.drop_duplicates()
        removed_duplicates = initial_rows - len(processed_data)

        if self.verbose and removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")

        # Validate that we still have data
        if len(processed_data) == 0:
            raise ValueError(
                "All data was removed during preprocessing. Check your dataset for quality issues."
            )

        if len(processed_data.columns) == 0:
            raise ValueError(
                "All features were removed during preprocessing. Check your dataset for quality issues."
            )

        return processed_data


    def _generate_synthetics_llm(
        self,
        train_data: pd.DataFrame,
        plugin_name: str = "gpt-oss:20b",
        n_samples: int = 100,
        batch_size: int = 10,
    ) -> pd.DataFrame:  
        """
        Generate synthetic data using LLM.

        Args:
            train_data (pd.DataFrame): Preprocessed training data.
            plugin_name (str): Plugin identifier, e.g. "gpt-oss:20b".
            n_samples (int): Number of synthetic samples to generate.
            batch_size (int): Number of samples to generate per batch.

        Returns:
            pd.DataFrame: Generated synthetic samples.
        """
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        columns = train_data.columns

        if self.verbose:
            print(f'Start generating data using model "{plugin_name}"')

        generator = LLMGenerator(
            gen_client=client,
            gen_model_nm=plugin_name,
            real_data=train_data,
            cols=columns,
            verbose=self.verbose,
        )

        generated_data = generator.generate(n_samples=n_samples, batch_size=batch_size)

        return generated_data


    def _generate_synthetics_non_llm(
        self,
        train_data: pd.DataFrame,
        plugin_name: str,
        task_type: str,
        optimization_trials: int = None,
        target_column: str = None,
        n_samples: int = 100,
        custom_metric: Metric = None,
        params: str = None,
        log_params: bool = False,
    ) -> pd.DataFrame:  
        """
        Generate synthetic data using non-LLM `synthcity` plugins with optional HPO.

        Args:
            train_data (pd.DataFrame): Preprocessed training data.
            plugin_name (str): Plugin identifier, e.g. "ctgan", "ddpm", "dpgan".
            optimization_trials (int): Number of Optuna trials for HPO.
            target_column (str, optional): Supervised target column, if any.
            n_samples (int): Target number of synthetic samples to generate.
            custom_metric (Metric, optional): Custom quality metric (reserved).
            params (str): Path to existing optimization parameters
            log_params (bool): If True, persist HPO studies to disk.

        Returns:
            pd.DataFrame: Generated synthetic samples.
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

        # Step 3: Training plugin (ctgan, ddpm or dpgan)
        if self.verbose and (optimization_trials != None or params != None):
            print(
                f'Start training model "{plugin_name}". Using parameters:\n{init_kwargs[plugin_name]}'
            )

        generators = Plugins()

        train_loader = GenericDataLoader(train_data, target_column=target_column)

        if plugin_name == "ddpm" and task_type == "classification":
            init_kwargs['ddpm']['is_classification'] = True

        # if init_kwargs[plugin_name] != {}:
        try:
            generator = generators.get(
                plugin_name,
                compress_dataset=False,
                strict=False,
                **init_kwargs[plugin_name],
            )
            generator.fit(train_loader)
            syn_df = generator.generate(n_samples)

        except ValueError:
            print("\nTrying fitting generator without optimization parameters...")
            try:
                generator = generators.get(
                    plugin_name,
                    compress_dataset=False,
                    strict=False,
                )
                generator.fit(train_loader)
                syn_df = generator.generate(n_samples)
            except Exception as e:
                print(
                    f"Synthetic data generation failed: {str(e)}. Please try another model"
                )

        return syn_df.dataframe()


    def _perform_curation(
        self,
        syn_data: pd.DataFrame,
        real_data: pd.DataFrame,
        n_generations: int = 20,
        crossover_prob: int | float = 0.6,
        bootstrap_sample_ratio: int | float = 0.9,
        target_column: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:

        config = GAConfig(
            n_generations=n_generations,
            crossover_prob=crossover_prob,
            bootstrap_sample_ratio=bootstrap_sample_ratio,
            verbose=verbose,
        )
        ga = GeneticAlgorithm(config=config, target_col=target_column)
        results = ga.run(syn_data, real_data)

        return results


    def _extract_outliers(self, df: pd.DataFrame, columns: Any = None, threshold=1.5):
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


    def generate(
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
        crossover_prob: int | float = 0.6,
        bootstrap_sample_ratio: int | float = 0.9,
    ):
        """
        Generate synthetic data based on the provided training data.

        This method loads and preprocesses the training data, then generates
        synthetic samples using the configured model and task settings.

        Args:
            train_data_path (str, optional): Path to the CSV containing training data.
            sep (str): CSV delimiter.
            n_samples (int): Number of synthetic samples to generate.
            batch_size (int): Number of samples to generate per batch for LLM.
            log_params (bool): If True, persist HPO studies to disk.
            custom_metric (Metric, optional): Placeholder for custom quality metric.
            optimization_trials (int): Number of trials for HPO.
            params (str): Path to existing optimization parameters.
            target_column (str, optional): Target column name for supervised setups.

        Returns:
            pd.DataFrame: Synthetic dataset.

        Raises:
            ValueError: If there's an error loading or preprocessing the data.

        Example:
            >>> synthesizer = TabAutoSyn(verbose=True)
            >>> syn_df = synthesizer.generate("data.csv", n_samples=500)
            >>> print(f"Generated {len(syn_df)} samples")
        """
        if train_data_path is not None:
            if n_samples is not None: 
                try:
                    # Load data with error handling
                    train_data = pd.read_csv(train_data_path, sep=sep)

                    if self.verbose:
                        print(
                            f"Loaded dataset with {len(train_data)} samples and {len(train_data.columns)} features"
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
                            optimization_trials=optimization_trials,
                            target_column=target_column,
                            n_samples=n_samples*1.5,
                            custom_metric=custom_metric,
                            params=params,
                            log_params=log_params,
                        )

                        syn_outliers = self._generate_synthetics_non_llm(
                            train_data=outliers,
                            plugin_name=plugin_name,
                            optimization_trials=optimization_trials,
                            target_column=target_column,
                            n_samples=len(outliers),
                            custom_metric=custom_metric,
                            params=params,
                            log_params=log_params,
                        )

                    elif self.model == "LLM":
                        syn_df = self._generate_synthetics_llm(
                            train_data=train_data_mod,
                            n_samples=n_samples*1.5,
                            batch_size=batch_size,
                        )

                        syn_outliers = self._generate_synthetics_llm(
                            train_data=outliers, n_samples=len(outliers), batch_size=batch_size
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

        if train_data_path is not None:
            try:
                # Load data with error handling
                train_data = pd.read_csv(train_data_path, sep=sep)

                if self.verbose:
                    print(
                        f"Loaded dataset with {len(train_data)} samples and {len(train_data.columns)} features"
                    )

                if run_preprocessing:
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
                            optimization_trials=optimization_trials,
                            target_column=target_column,
                            n_samples=len(outliers),
                            custom_metric=custom_metric,
                            params=params,
                            log_params=log_params,
                        )

                        syn_outliers = syn_outliers.dropna()
                        if syn_outliers.duplicated().any():
                            syn_outliers = syn_outliers.drop_duplicates()

                        return syn_outliers

                    elif self.model == "LLM":
                        syn_outliers = self._generate_synthetics_llm(
                            train_data=outliers,
                            n_samples=len(outliers),
                            batch_size=batch_size,
                            )

                        syn_outliers = syn_outliers.dropna()
                        if syn_outliers.duplicated().any():
                            syn_outliers = syn_outliers.drop_duplicates()
                            
                        return syn_outliers

                else:
                    if n_samples is not None:
                        if self.model == "task_specific":
                            syn_df = self._generate_synthetics_non_llm(
                                train_data=(
                                    train_data_mod if run_preprocessing else train_data
                                ),
                                plugin_name=plugin_name,
                                optimization_trials=optimization_trials,
                                target_column=target_column,
                                n_samples=n_samples*1.5,
                                custom_metric=custom_metric,
                                params=params,
                                log_params=log_params,
                            )

                            syn_df = syn_df.dropna()
                            if syn_df.duplicated().any():
                                syn_df = syn_df.drop_duplicates()
                                
                            return syn_df

                        elif self.model == "LLM":
                            syn_df = self._generate_synthetics_llm(
                                train_data=(
                                    train_data_mod if run_preprocessing else train_data
                                ),
                                n_samples=n_samples*1.5,
                                batch_size=batch_size,
                            )

                            syn_df = syn_df.dropna()
                            if syn_df.duplicated().any():
                                syn_df = syn_df.drop_duplicates()
                                
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
        crossover_prob: int | float = 0.6,
        bootstrap_sample_ratio: int | float = 0.9,
        target_column: str = None,
    ):

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
