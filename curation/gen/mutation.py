import random
from gen.individ import Individual
from abc import ABC, abstractmethod
from typing import List, Tuple


class MutationOperator(ABC):
    """Abstract class for mutation operator"""

    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        """Perform mutation"""
        pass


class ReplacementMutation(MutationOperator):
    """Mutation by replacing rows from global pool"""

    def __init__(self, global_pool: List[Tuple], mutation_rate: float = 0.05):
        self.global_pool = global_pool
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual) -> Individual:
        """Replace random rows of individual"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.global_pool)
        return individual
