# Particle Swarm Optimzation
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from knapsack import Problem
from knapsack import Solution as BaseSolution


@dataclass
class Solution(BaseSolution):
    velocity: np.ndarray


@dataclass
class PSO:  # Particle Swarm Optimzation
    problem: Problem
    max_iterations: int
    population: np.ndarray
    N: int
    best: Solution
    alpha: float  # Proportion of velocity to be retained
    beta: float  # Proportion of personal best to be retained
    delta: float  # Proportion of global best to be retained
    epsilon: float  # jump size of particle
    v_max: float

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
        velocity = np.random.uniform(-self.v_max, self.v_max, self.problem.size)
        return Solution(cells=cells, fitness=fitness, weight=weight, velocity=velocity)

    def init_population(self) -> None:
        for i in range(self.N):
            cells = np.random.randint(0, 2, self.problem.size)
            self.population[i] = self.repair(cells)

    def set_global_best(self):
        self.population = np.array(sorted(self.population, reverse=True), dtype=object)
        if self.population[0] > self.best:
            self.best = deepcopy(self.population[0])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binarization(self, X: np.ndarray) -> np.ndarray:
        X = self.sigmoid(X)
        rnd = np.random.uniform(0, 1, self.problem.size)
        X[X < rnd] = 0
        X[X >= rnd] = 1
        return X

    def solve(self):
        self.init_population()
        self.set_global_best()
        for _ in range(self.max_iterations):
            best_local = self.population[0]
            for i in range(self.N):
                # Update Velocity
                self.population[i].velocity = (
                    self.alpha * self.population[i].velocity
                    + np.random.uniform(0, self.beta)
                    * (self.best.cells - self.population[i].cells)
                    + np.random.uniform(0, self.delta)
                    * (best_local.cells - self.population[i].cells)
                )

                # Update cells
                cells = (
                    self.population[i].cells
                    + self.epsilon * self.population[i].velocity
                )
                self.population[i].cells = self.binarization(cells)
                self.population[i] = self.repair(self.population[i].cells)
            self.set_global_best()
        return self.best
