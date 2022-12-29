# Grey Wolf Optimizer
from dataclasses import dataclass

import numpy as np

from knapsack import Problem, Solution


@dataclass
class GWO:  # Grey Wolf Optimizer
    max_iterations: int
    N: int
    problem: Problem
    population: np.ndarray
    a: float
    alpha: Solution
    beta: Solution
    delta: Solution

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

    def init_wolf(self):
        cells = np.random.randint(2, size=self.problem.size)
        return self.repair(cells)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binarization(self, X: np.ndarray) -> np.ndarray:
        X = self.sigmoid(X)
        rnd = np.random.uniform(0, 1, self.problem.size)
        X[X < rnd] = 0
        X[X >= rnd] = 1
        return X

    def update_alpha_beta_delta(self):
        self.population = np.array(
            sorted(self.population, reverse=True, key=lambda x: x.fitness),
            dtype=object,
        )
        self.alpha = self.population[0]
        self.beta = self.population[1]
        self.delta = self.population[2]

    def update_population(self):
        for i in range(3, self.N):
            A1 = 2 * self.a * np.random.uniform(0, 1, self.problem.size) - self.a
            C1 = 2 * np.random.uniform(0, 1, self.problem.size)
            D_alpha = np.abs(C1 * self.alpha.cells - self.population[i].cells)
            X1 = self.alpha.cells - A1 * D_alpha

            A2 = 2 * self.a * np.random.uniform(0, 1, self.problem.size) - self.a
            C2 = 2 * np.random.uniform(0, 1, self.problem.size)
            D_beta = np.abs(C2 * self.beta.cells - self.population[i].cells)
            X2 = self.beta.cells - A2 * D_beta

            A3 = 2 * self.a * np.random.uniform(0, 1, self.problem.size) - self.a
            C3 = 2 * np.random.uniform(0, 1, self.problem.size)
            D_delta = np.abs(C3 * self.delta.cells - self.population[i].cells)
            X3 = self.delta.cells - A3 * D_delta

            X = (X1 + X2 + X3) / 3
            X = self.binarization(X)
            solution = self.repair(X)
            self.population[i] = solution

    def solve(self) -> Solution:
        self.population = np.array([self.init_wolf() for _ in range(self.N)])
        it = 0
        while it < self.max_iterations:
            self.a = 2 - it * ((2) / self.max_iterations)
            self.update_alpha_beta_delta()
            self.update_population()
            it += 1
        self.update_alpha_beta_delta()
        return self.alpha
