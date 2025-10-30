from tabnanny import verbose
import pandas as pd
from tabautosyn.custom_metric import Metric  ### in progress ###
from tabautosyn.optimization import HyperparameterOptimizer
from tabautosyn.outliers import ExtractOutliers  ### in progress ###
from tabautosyn.llm_generator import LLMGenerator
from tabautosyn.gen.gen import GAConfig, GeneticAlgorithm

# Synthcity
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

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
        batch_size: int = 10) -> pd.DataFrame: ### in progress ####
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
        client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key ='ollama'
        )

        columns = train_data.columns

        if self.verbose:
            print(f'Start generating data using model "{plugin_name}"')

        generator = LLMGenerator(gen_client=client, gen_model_nm=plugin_name, real_data=train_data, cols=columns, verbose=self.verbose)

        generated_data = generator.generate(n_samples=n_samples, batch_size=batch_size)

        return generated_data


    def _generate_synthetics_non_llm(
        self,
        train_data: pd.DataFrame,
        plugin_name: str,
        optimization_trials: int = None,
        target_column: str = None,
        n_samples: int = 100,
        custom_metric: Metric = None,
        log_params: bool = False,
    ) -> pd.DataFrame:  ### in progress ####
        """
        Generate synthetic data using non-LLM `synthcity` plugins with optional HPO.

        Args:
            train_data (pd.DataFrame): Preprocessed training data.
            plugin_name (str): Plugin identifier, e.g. "ctgan", "ddpm", "dpgan".
            optimization_trials (int): Number of Optuna trials for HPO.
            target_column (str, optional): Supervised target column, if any.
            n_samples (int): Target number of synthetic samples to generate.
            custom_metric (Metric, optional): Custom quality metric (reserved).
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
        if optimization_trials != None:
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
                    print(f"Best score for {plugin_name}: {optimization_result.best_value}")

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Hyperparameter optimization failed: {str(e)}")

        # Step 2: Training plugin (ctgan, ddpm or dpgan)
        if self.verbose:
            print(f'Start training model "{plugin_name}"')

        generators = Plugins()

        train_loader = GenericDataLoader(train_data, target_column=target_column)

        # if plugin_name == "ddpm" and ml_task == "classification":
        #     init_kwargs['ddpm']['is_classification'] = True

        if optimization_trials != None:
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
                **init_kwargs[plugin_name],
            )

        generator.fit(train_loader)

        syn_df = generator.generate(n_samples * 2)

        return syn_df.dataframe()

    
    def _perform_curation(
        self, 
        syn_data: pd.DataFrame, 
        real_data: pd.DataFrame,
        target_column: str = None,
        verbose: bool = False
        ) -> pd.DataFrame:

        config = GAConfig(n_generations=20, crossover_prob=0.6, bootstrap_sample_ratio=0.9, verbose=verbose)
        ga = GeneticAlgorithm(config=config, target_col=target_column)
        results = ga.run(syn_data, real_data)

        return results


    def generate(
        self,
        train_data_path: str = None,
        sep: str = ",",
        n_samples: int = 100,
        batch_size: int = 10,
        log_params: bool = False,
        custom_metric: Metric = None,
        optimization_trials: int = None,
        target_column: str = None,
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
                if optimization_trials != None:
                    if self.verbose and self.model != "LLM":
                        print("Starting hyperparameter optimization...")

                if self.model == "task_specific":
                    if self.task == "privacy":
                        plugin_name = "dpgan"
                    elif self.task == "ml":
                        plugin_name = "ddpm"
                    elif self.task == "universal":
                        plugin_name = "ctgan"

                # We need to extract outliers to make better distribution (### in progress ####)
                # extractor = ExtractOutliers()
                # outliers = extractor._extract_outliers(train_data=train_data_mod)

                if self.model == "task_specific":
                    syn_df_full = self._generate_synthetics_non_llm(
                        train_data=train_data_mod,
                        plugin_name=plugin_name,
                        optimization_trials=optimization_trials,
                        target_column=target_column,
                        n_samples=n_samples,
                        custom_metric=custom_metric,
                        log_params=log_params,
                    )

                    # syn_outliers = self._generate_synthetics_non_llm(train_data=outliers,
                    #                                                 plugin_name=plugin_name,
                    #                                                 optimization_trials=optimization_trials,
                    #                                                 target_column=target_column,
                    #                                                 n_samples=n_samples,
                    #                                                 custom_metric=custom_metric,
                    #                                                 log_params=log_params)
        
                    # pre_curated_df = pd.concat([syn_df_full, syn_outliers], axis=1)

                    # Start curation process
                    if self.verbose:
                        print(f'Starting evolutional optimization ...')

                    final_curated_df = self._perform_curation(syn_data=syn_df_full, real_data=train_data_mod, target_column=target_column, verbose=self.verbose)

                    return final_curated_df

                elif self.model == "LLM":
                    syn_df_full = self._generate_synthetics_llm(
                        train_data=train_data_mod,
                        n_samples=n_samples,
                        batch_size=batch_size,
                    )
                #     syn_outliers = self._generate_synthetics_llm(outliers)

            except Exception as e:
                raise ValueError(f"Error loading or preprocessing data: {str(e)}")

        return syn_df_full
