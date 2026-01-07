import random
from typing import List
from .individ import Individual
from abc import ABC, abstractmethod


class SelectionOperator(ABC):
    """Abstract class for selection operator"""

    @abstractmethod
    def select(self, population: List[Individual], n: int) -> List[Individual]:
        """Select individuals from population"""
        pass


class TournamentSelection(SelectionOperator):
    """Tournament selection"""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: List[Individual], n: int) -> List[Individual]:
        """Tournament selection"""
        selected = []
        for _ in range(n):
            tournament = random.sample(population, self.tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness_value or -float("inf"))
            selected.append(winner)
        return selected
