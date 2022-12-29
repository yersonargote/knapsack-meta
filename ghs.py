# Global Harmony Search
from dataclasses import dataclass

import numpy as np

from knapsack import Problem, Solution


@dataclass
class GHS:
    problem: Problem
    memory: np.ndarray
    max_iterations: int
    N: int
    HMCR: float
    PAR: float

    def repair(self, cells: np.ndarray) -> Solution:
        weight = self.problem.weigh(cells)
        if weight > self.problem.capacity:
            for i in np.random.permutation(np.arange(self.problem.size)):
                if cells[i] == 1 and weight > self.problem.capacity:
                    cells[i] = 0
                    weight -= self.problem.weights[i]
        else:
            for i in np.random.permutation(np.arange(self.problem.size)):
                if (
                    cells[i] == 0
                    and self.problem.weights[i] + weight <= self.problem.capacity
                ):
                    cells[i] = 1
                    weight += self.problem.weights[i]
        fitness = self.problem.evaluate(cells)
        return Solution(cells=cells, fitness=fitness, weight=weight)

    def init_random_memory(self) -> None:
        for i in range(self.N):
            cells = np.random.randint(0, 2, self.problem.size)
            self.memory[i] = self.repair(cells)
        self.memory = np.array(sorted(self.memory, reverse=True))

    def update_memory(self, solution):
        if solution.fitness > self.memory[-1].fitness:
            self.memory[-1] = solution
        self.memory = np.array(sorted(self.memory, reverse=True))

    def solve(self) -> Solution:
        self.init_random_memory()
        for _ in range(self.max_iterations):
            cells = np.zeros(self.problem.size)
            for j in range(self.problem.size):
                rnd = np.random.uniform()
                if rnd <= self.HMCR:
                    x = np.random.randint(self.N)
                    cells[j] = self.memory[x].cells[j]
                    rnd = np.random.uniform()
                    if rnd <= self.PAR:
                        cells[j] = self.memory[0].cells[j]
                else:
                    cells[j] = np.random.randint(0, 2)
            solution = self.repair(cells)
            self.update_memory(solution)
        return self.memory[0]
