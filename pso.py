# Particle Swarm Optimzation
import numpy as np
from knapsack import Problem
from knapsack import Solution as BaseSolution
from dataclasses import dataclass


@dataclass
class Solution(BaseSolution):
    velocity: np.ndarray


@dataclass
class PSO: # Particle Swarm Optimzation
    problem: Problem
    max_iterations: int
    population: np.ndarray
    N: int
    omega: float
    phi_p: float  # cognitive/local weight
    phi_g: float  # social/global weight
    v_max: float
    global_best: Solution

    def repair(self, cells: np.ndarray) -> Solution:
        weight = self.problem.weigh(cells)
        if weight > self.problem.capacity:
            for i in np.random.permutation(np.arange(self.problem.size)):
                if cells[i] == 1 and weight > self.problem.capacity:
                    cells[i] = 0
                    weight -= self.problem.weights[i]
        else:
            for i in np.random.permutation(np.arange(self.problem.size)):
                if cells[i] == 0 and self.problem.weights[i] + weight <= self.problem.capacity:
                    cells[i] = 1
                    weight += self.problem.weights[i]
        fitness = self.problem.evaluate(cells)
        velocity = np.random.uniform(0, self.v_max, self.problem.size)
        return Solution(cells=cells, fitness=fitness, weight=weight, velocity=velocity)

    def init_population(self) -> None:
        for i in range(self.N):
          cells = np.random.randint(0, 2, self.problem.size)
          self.population[i] = self.repair(cells)

    def get_global_best(self):
        return np.array(sorted(self.population, reverse=True))[0]

    def update_velocity(self, solution: Solution) -> Solution:
        r_p, r_g = np.random.uniform(0, 1, size=2)
        vel_cognitive = self.phi_p * r_p * (solution.cells - solution.velocity)
        vel_social = self.phi_g * r_g * (self.global_best.cells - solution.cells)
        solution.velocity = self.omega * solution.velocity + vel_cognitive + vel_social
        return solution

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binarization(self, X: np.ndarray) -> np.ndarray:
        X = self.sigmoid(X)
        rnd = np.random.uniform(0, 1, self.problem.size)
        X[X < rnd] = 0
        X[X >= rnd] = 1
        return X

    def update_position(self, solution: Solution) -> Solution:
        cells = solution.cells + solution.velocity
        cells = self.binarization(cells)
        solution = self.repair(cells)
        return solution

    def solve(self):
        self.init_population()
        self.global_best = self.get_global_best()
        for _ in range(self.max_iterations):
            for i in range(self.N):
                solution = self.population[i]
                solution = self.update_velocity(solution)
                solution = self.update_position(solution)
                self.population[i] = solution
            self.global_best = self.get_global_best()
        return self.global_best
