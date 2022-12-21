from dataclasses import dataclass

import numpy as np


# Knapsack
@dataclass
class Problem:
    name: str
    size: int
    capacity: int
    weights: np.ndarray
    profits: np.ndarray
    optimal: float

    def evaluate(self, cells: np.ndarray) -> int:
        return np.dot(cells, self.profits)

    def weigh(self, cells: np.ndarray) -> int:
        return np.dot(cells, self.weights)


@dataclass
class Solution:
    cells: np.ndarray
    fitness: float
    weight: float

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __ge__(self, other):
        return self.fitness > other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness
