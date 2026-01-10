import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass

# from deap import base, creator, tools
from .individ import Individual
from .fitness import FitnessEvaluator, MLFitnessEvaluator
from .crossover import CrossoverOperator, ExchangeCrossover,UniqueExchangeCrossover
from .mutation import MutationOperator, ReplacementMutation
from .selection import SelectionOperator, TournamentSelection

from rich.console import Console

import logging

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Genetic algorithm configuration
    Args:
        n_generations: number of generations
        crossover_prob: probability of applying crossover
        mutation_prob: probability of mutation
        tournament_size: number of individuals participating in tournament selection
        population_size: population size
        n_bootstrap_samples: initial population size when using bootstrap
        bootstrap_sample_ratio: bootstrap sample ratio

    """

    n_generations: int = 50
    crossover_prob: float = 1.0
    mutation_prob: float = 0.02
    tournament_size: int = 3
    population_size: Optional[int] = None
    verbose: bool = True
    n_bootstrap_samples: int = 50
    bootstrap_sample_ratio: list = None
    exchange_rate_range: Tuple[float, float] = (0.05, 0.3)
    mutation_rate: float = 0.05


@dataclass
class GAResult:
    """Genetic algorithm result"""

    rank: int
    fitness: float
    n_rows: int
    df: pd.DataFrame


class GeneticAlgorithm:
    """Main genetic algorithm class
    Args:
        config: configuration
        fitness_evaluator: fitness function
        crossover_operator: crossover operator
        mutation_operator: mutation operator
        selection_operator: selection operator
        target_col: target variable

    """

    def __init__(
        self,
        config: Optional[GAConfig] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        crossover_operator: Optional[CrossoverOperator] = None,
        mutation_operator: Optional[MutationOperator] = None,
        selection_operator: Optional[SelectionOperator] = None,
        target_col: str = "target",
    ):
        self.config = config or GAConfig()
        self.target_col = target_col

        # Default components
        self.fitness_evaluator = fitness_evaluator or MLFitnessEvaluator(target_col)
        self.crossover_operator = crossover_operator or UniqueExchangeCrossover(
            exchange_rate_range=self.config.exchange_rate_range
        )
        self.selection_operator = selection_operator or TournamentSelection(
            tournament_size=self.config.tournament_size
        )

        # mutation_operator will be initialized in run() when global_pool is known
        self.mutation_operator = mutation_operator

        self.history = {"max": [], "avg": []}
        self.global_pool = None
        self.feature_cols = None

    def _evaluate_population(
        self, population: List[Individual], test_data: pd.DataFrame
    ):
        """Evaluate entire population"""
        for ind in population:
            if ind.fitness_value is None:
                ind.fitness_value = self.fitness_evaluator.evaluate(ind, test_data)

    def _evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Evolve one generation"""
        # Selection
        offspring = self.selection_operator.select(population, len(population))
        offspring = [
            Individual(list(ind.data), ind.feature_cols, ind.target_col)
            for ind in offspring
        ]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.config.crossover_prob:
                offspring[i], offspring[i + 1] = self.crossover_operator.crossover(
                    offspring[i], offspring[i + 1]
                )
                offspring[i].fitness_value = None
                offspring[i + 1].fitness_value = None

        # Mutation
        for ind in offspring:
            if random.random() < self.config.mutation_prob:
                self.mutation_operator.mutate(ind)
                ind.fitness_value = None

        return offspring

    def run(
        self,
        syn_data: pd.DataFrame,
        real_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        initial_population: Optional[List[pd.DataFrame]] = None,
    ) -> List[GAResult]:
        """Run genetic algorithm

        Args:
            syn_data: Training data
            real_data: Test data
            feature_cols: List of feature columns (if None, all except target_col)
            initial_population: Initial population (if None, created via bootstrap)
        """

        # Determine feature columns
        if feature_cols is None:
            feature_cols = [c for c in syn_data.columns if c != self.target_col]
        self.feature_cols = feature_cols

        syn_data, real_data = filter_rare_classes(
            syn_data, real_data, target_col=self.target_col
        )

        # Create initial population via bootstrap if not provided
        if initial_population is None:

            max_attempts = 5
            target_classes = set(syn_data[self.target_col].unique())
            initial_population = None
            valid_population = False

            for attempt in range(1, max_attempts + 1):
                initial_population = random_subsampling(
                    syn_data,
                    n_samples=self.config.n_bootstrap_samples,
                    sample_ratio=self.config.bootstrap_sample_ratio,
                )

                # Ensure every bootstrap sample includes all target classes
                if all(
                    target_classes.issubset(set(df[self.target_col].unique()))
                    for df in initial_population
                ):
                    valid_population = True
                    if self.config.verbose:
                        logger.info(
                            f"Initial population successfully created on attempt {attempt}."
                        )
                    break
                else:
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts}: not all target classes are present in bootstrap samples."
                    )

            if not valid_population:
                logger.error(
                    f"Failed to create initial_population with all target classes after {max_attempts} attempts. "
                    "Continuing with the last generated population."
                )

        # Create global pool for mutation
        if self.global_pool is None:
            self.global_pool = create_global_pool(
                initial_population, feature_cols, self.target_col
            )

        # Initialize mutation operator if not provided
        if self.mutation_operator is None:
            self.mutation_operator = ReplacementMutation(
                self.global_pool, mutation_rate=self.config.mutation_rate
            )

        # Create initial population from Individual
        population = [
            Individual.from_dataframe(df, feature_cols, self.target_col)
            for df in initial_population
        ]

        # Initial evaluation
        self._evaluate_population(population, real_data)

        # Evolution
        if self.config.verbose:
            console = Console()
            with console.status("[bold green]Working on tasks...") as status:
                for gen in range(1, self.config.n_generations + 1):
                    population = self._evolve_generation(population)
                    self._evaluate_population(population, real_data)

                    # Statistics
                    fitness_values = [ind.fitness_value for ind in population]
                    self.history["max"].append(max(fitness_values))
                    self.history["avg"].append(np.mean(fitness_values))

                    if self.config.verbose:
                        console.log(
                            f"Gen {gen}: max={self.history['max'][-1]:.4f}, avg={self.history['avg'][-1]:.4f}"
                        )

        # Results
        population.sort(key=lambda ind: ind.fitness_value, reverse=True)

        results = []
        for rank, ind in enumerate(population[:1], start=1):
            results.append(
                GAResult(
                    rank=rank,
                    fitness=ind.fitness_value,
                    n_rows=len(ind),
                    df=ind.to_dataframe(),
                )
            )

        return results[0].df

    def plot_history(self):
        """Visualize algorithm progress"""
        plt.figure(figsize=(10, 5))
        generations = range(1, len(self.history["max"]) + 1)
        plt.plot(generations, self.history["max"], label="Max Fitness", linewidth=2)
        plt.plot(generations, self.history["avg"], label="Avg Fitness", linewidth=2)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        plt.title("Genetic Algorithm Progress", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def bootstrap_sample(
    data: pd.DataFrame, n_samples: int = 1000, sample_ratio: float = 0.7
) -> List[pd.DataFrame]:
    """Create bootstrap samples"""
    bootstrap_samples = []
    n = int(len(data) * sample_ratio)
    for _ in range(n_samples):
        indices = np.random.randint(0, len(data), size=n)
        sample = data.iloc[indices].reset_index(drop=True)
        bootstrap_samples.append(sample)
    return bootstrap_samples

def random_subsampling(data: pd.DataFrame, n_samples: int = 10, 
                     sample_ratios: list = None):
    
    if sample_ratios is None:
        sample_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    bootstrap_samples = {}
    
    for ratio in sample_ratios:
        n = int(len(data) * ratio)
        samples = []
        for _ in range(n_samples):
            indices = np.random.choice(len(data), size=n, replace=False)
            sample = data.iloc[indices].reset_index(drop=True)
            samples.append(sample)
        bootstrap_samples[ratio] = samples
   
    return bootstrap_samples


def create_global_pool(
    dataframes: List[pd.DataFrame], feature_cols: List[str], target_col: str
) -> List[Tuple]:
    """Create global pool of unique rows"""
    global_pool = set()
    for df in dataframes:
        for row in df[feature_cols + [target_col]].itertuples(index=False, name=None):
            global_pool.add(tuple(row))

    return list(global_pool)


def filter_rare_classes(df_train, df_test, target_col):
    """
    Return:
    ------------
    df_train_filtered, df_test_filtered : (pd.DataFrame, pd.DataFrame)

    """
    min_count = len(df_train) * 0.015
    class_counts = df_train[target_col].value_counts()

    valid_classes = class_counts[class_counts >= min_count].index

    df_train_filtered = df_train[df_train[target_col].isin(valid_classes)].copy()
    df_test_filtered = df_test[df_test[target_col].isin(valid_classes)].copy()

    return df_train_filtered, df_test_filtered

