import random
from typing import Tuple
from gen.individ import Individual
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
    """Crossover by exchanging rows (without duplicates)"""

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
            row1, row2 = ind1[a], ind2[b]

            # Avoid duplicates
            if row2 in ind1.data:
                alternatives = [r for r in ind2.data if r not in ind1.data]
                if alternatives:
                    row2 = random.choice(alternatives)

            if row1 in ind2.data:
                alternatives = [r for r in ind1.data if r not in ind2.data]
                if alternatives:
                    row1 = random.choice(alternatives)

            ind1[a], ind2[b] = row2, row1

        return ind1, ind2
