import random
from typing import Tuple
from .individ import Individual
from abc import ABC, abstractmethod


class CrossoverOperator(ABC):
    """Abstract class for crossover operator"""

    @abstractmethod
    def crossover(
        self, ind1: Individual, ind2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform crossover"""
        pass


class ExchangeCrossover(CrossoverOperator):
    """Crossover by exchanging rows (with duplicates)"""

    def __init__(self, exchange_rate_range: Tuple[float, float] = (0.05, 0.3)):
        self.exchange_rate_range = exchange_rate_range

    def crossover(
        self, ind1: Individual, ind2: Individual
    ) -> Tuple[Individual, Individual]:
        """Exchange rows between two individuals"""
        size = min(len(ind1), len(ind2))
        k = max(1, int(size * random.uniform(*self.exchange_rate_range)))

        idxs1 = random.sample(range(size), k)
        idxs2 = random.sample(range(size), k)

        for a, b in zip(idxs1, idxs2):

            ind1[a], ind2[b] = ind2[b], ind1[a]

        return ind1, ind2


class UniqueExchangeCrossover(CrossoverOperator):
    """Crossover by exchanging rows (without duplicates)"""

    def __init__(self, exchange_rate_range: Tuple[float, float] = (0.05, 0.3)):
        self.exchange_rate_range = exchange_rate_range

    def crossover(
        self, ind1: Individual, ind2: Individual
    ) -> Tuple[Individual, Individual]:
        """Exchange unique rows between two individuals while preserving sizes and avoiding duplicates"""

        data1 = list(ind1.data)
        data2 = list(ind2.data)

        set_ind2 = set(data2)
        set_ind1 = set(data1)

        unique_in_ind1 = [
            (i, row) for i, row in enumerate(data1) if row not in set_ind2
        ]
        unique_in_ind2 = [
            (i, row) for i, row in enumerate(data2) if row not in set_ind1
        ]

        if not unique_in_ind1 or not unique_in_ind2:
            offspring1 = Individual(data1, ind1.feature_cols, ind1.target_col)
            offspring2 = Individual(data2, ind2.feature_cols, ind2.target_col)
            return offspring1, offspring2

        idx1, row1 = random.choice(unique_in_ind1)
        idx2, row2 = random.choice(unique_in_ind2)

        data1[idx1] = row2
        data2[idx2] = row1

        offspring1 = Individual(data1, ind1.feature_cols, ind1.target_col)
        offspring2 = Individual(data2, ind2.feature_cols, ind2.target_col)

        return offspring1, offspring2
