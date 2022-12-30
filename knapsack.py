from dataclasses import dataclass

import numpy as np


@dataclass
class Problem:
    """Binary knapsack problem"""

    name: str
    size: int
    capacity: int
    weights: np.ndarray
    profits: np.ndarray
    optimal: float

    def evaluate(self, cells: np.ndarray) -> int:
        """Calculate the profit of the cells"""
        return np.dot(cells, self.profits)

    def weigh(self, cells: np.ndarray) -> int:
        """Calculate the weigh of the cells"""
        return np.dot(cells, self.weights)


@dataclass
class Solution:
    """Posible solution for knapsack problem"""

    cells: np.ndarray
    fitness: float
    weight: float

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __ge__(self, other):
        return self.fitness > other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness
