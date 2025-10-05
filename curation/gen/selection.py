import random
from typing import List
from gen.individ import Individual
from abc import ABC, abstractmethod


class SelectionOperator(ABC):
    """Абстрактный класс для оператора селекции"""

    @abstractmethod
    def select(self, population: List[Individual], n: int) -> List[Individual]:
        """Выбор индивидов из популяции"""
        pass


class TournamentSelection(SelectionOperator):
    """Турнирная селекция"""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: List[Individual], n: int) -> List[Individual]:
        """Турнирная селекция"""
        selected = []
        for _ in range(n):
            tournament = random.sample(population, self.tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness_value or -float("inf"))
            selected.append(winner)
        return selected
