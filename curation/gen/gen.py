import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
from deap import base, creator, tools
from gen.individ import Individual
from gen.fitness import FitnessEvaluator, MLFitnessEvaluator
from gen.crossover import CrossoverOperator, ExchangeCrossover
from gen.mutation import MutationOperator, ReplacementMutation
from gen.selection import SelectionOperator, TournamentSelection


@dataclass
class GAConfig:
    """Конфигурация генетического алгоритма
    Args:
        n_generations: количество поколений
        crossover_prob: вероятность применения кроссовера
        mutation_prob: вероятность мутации
        tournament_size: количество особей, участвующие в турнирном отборе
        population_size: размер популяции
        n_bootstrap_samples: размер начальной популяции при использовании бутстрепа
        bootstrap_sample_ratio:

    """

    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.02
    tournament_size: int = 3
    population_size: Optional[int] = None
    verbose: bool = True
    n_bootstrap_samples: int = 50
    bootstrap_sample_ratio: float = 0.7
    exchange_rate_range: Tuple[float, float] = (0.05, 0.3)
    mutation_rate: float = 0.05


@dataclass
class GAResult:
    """Результат работы генетического алгоритма"""

    rank: int
    fitness: float
    n_rows: int
    df: pd.DataFrame


class GeneticAlgorithm:
    """Основной класс генетического алгоритма
    Args:
        config:
        fitness_evaluator: fitness-функция
        crossover_operator: оператор кроссовера
        mutation_operator: оператор мутации
        selection_operator: оператор отбора
        target_col: целевая переменная

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

        # Компоненты по умолчанию
        self.fitness_evaluator = fitness_evaluator or MLFitnessEvaluator(target_col)
        self.crossover_operator = crossover_operator or ExchangeCrossover(
            exchange_rate_range=self.config.exchange_rate_range
        )
        self.selection_operator = selection_operator or TournamentSelection(
            tournament_size=self.config.tournament_size
        )

        # mutation_operator будет инициализирован в run() когда будет известен global_pool
        self.mutation_operator = mutation_operator

        self.history = {"max": [], "avg": []}
        self.global_pool = None
        self.feature_cols = None

    def _evaluate_population(
        self, population: List[Individual], test_data: pd.DataFrame
    ):
        """Оценка всей популяции"""
        for ind in population:
            if ind.fitness_value is None:
                ind.fitness_value = self.fitness_evaluator.evaluate(ind, test_data)

    def _evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Эволюция одного поколения"""
        # Селекция
        offspring = self.selection_operator.select(population, len(population))
        offspring = [
            Individual(list(ind.data), ind.feature_cols, ind.target_col)
            for ind in offspring
        ]

        # Кроссовер
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.config.crossover_prob:
                offspring[i], offspring[i + 1] = self.crossover_operator.crossover(
                    offspring[i], offspring[i + 1]
                )
                offspring[i].fitness_value = None
                offspring[i + 1].fitness_value = None

        # Мутация
        for ind in offspring:
            if random.random() < self.config.mutation_prob:
                self.mutation_operator.mutate(ind)
                ind.fitness_value = None

        return offspring

    def run(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        initial_population: Optional[List[pd.DataFrame]] = None,
    ) -> List[GAResult]:
        """Запуск генетического алгоритма

        Args:
            df_train: Тренировочные данные
            df_test: Тестовые данные
            feature_cols: Список колонок признаков (если None, все кроме target_col)
            initial_population: Начальная популяция (если None, создается через bootstrap)
        """

        # Определение колонок признаков
        if feature_cols is None:
            feature_cols = [c for c in df_train.columns if c != self.target_col]
        self.feature_cols = feature_cols

        # Создание начальной популяции через bootstrap, если не предоставлена
        if initial_population is None:
            initial_population = bootstrap_sample(
                df_train,
                n_samples=self.config.n_bootstrap_samples,
                sample_ratio=self.config.bootstrap_sample_ratio,
            )

        # Создание глобального пула для мутации
        if self.global_pool is None:
            self.global_pool = create_global_pool(
                initial_population, feature_cols, self.target_col
            )

        # Инициализация оператора мутации, если не был предоставлен
        if self.mutation_operator is None:
            self.mutation_operator = ReplacementMutation(
                self.global_pool, mutation_rate=self.config.mutation_rate
            )

        # Создание начальной популяции из Individual
        population = [
            Individual.from_dataframe(df, feature_cols, self.target_col)
            for df in initial_population
        ]

        # Начальная оценка
        self._evaluate_population(population, df_test)

        # Эволюция
        for gen in range(1, self.config.n_generations + 1):
            population = self._evolve_generation(population)
            self._evaluate_population(population, df_test)

            # Статистика
            fitness_values = [ind.fitness_value for ind in population]
            self.history["max"].append(max(fitness_values))
            self.history["avg"].append(np.mean(fitness_values))

            if self.config.verbose:
                print(
                    f"Gen {gen}: max={self.history['max'][-1]:.4f} "
                    f"avg={self.history['avg'][-1]:.4f}"
                )

        # Результаты
        population.sort(key=lambda ind: ind.fitness_value, reverse=True)

        results = []
        for rank, ind in enumerate(population[:10], start=1):
            results.append(
                GAResult(
                    rank=rank,
                    fitness=ind.fitness_value,
                    n_rows=len(ind),
                    df=ind.to_dataframe(),
                )
            )

        return results

    def plot_history(self):
        """Визуализация прогресса алгоритма"""
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
    """Создание bootstrap выборок"""
    bootstrap_samples = []
    n = int(len(data) * sample_ratio)
    for _ in range(n_samples):
        indices = np.random.randint(0, len(data), size=n)
        sample = data.iloc[indices].reset_index(drop=True)
        bootstrap_samples.append(sample)
    return bootstrap_samples


def create_global_pool(
    dataframes: List[pd.DataFrame], feature_cols: List[str], target_col: str
) -> List[Tuple]:
    """Создание глобального пула уникальных строк"""
    global_pool = set()
    for df in dataframes:
        for row in df[feature_cols + [target_col]].itertuples(index=False, name=None):
            global_pool.add(tuple(row))
    return list(global_pool)
