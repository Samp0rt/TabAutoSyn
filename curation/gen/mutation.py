import random
from gen.individ import Individual
from abc import ABC, abstractmethod
from typing import List, Tuple

class MutationOperator(ABC):
    """Абстрактный класс для оператора мутации"""
    
    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        """Выполнение мутации"""
        pass


class ReplacementMutation(MutationOperator):
    """Мутация через замену строк из глобального пула"""
    
    def __init__(self, global_pool: List[Tuple], mutation_rate: float = 0.05):
        self.global_pool = global_pool
        self.mutation_rate = mutation_rate
    
    def mutate(self, individual: Individual) -> Individual:
        """Замена случайных строк индивида"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.global_pool)
        return individual