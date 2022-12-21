# Genetic Algorithm
import numpy as np
from knapsack import Problem, Solution
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GA:
    N: int
    generations: int
    problem: Problem
    population: np.ndarray
    opponents: int

    def init_individual(self) -> Solution:
      cells = np.random.randint(0, 2, self.problem.size)
      weight = self.problem.weigh(cells)
      while weight > self.problem.capacity:
        ones = np.where(cells == 1)[0]
        takeout = np.random.choice(ones)
        cells[takeout] = 0
        weight -= self.problem.weights[takeout]
      fitness = self.problem.evaluate(cells)
      individual = Solution(cells=cells, fitness=fitness, weight=weight,)
      return individual

    def init_population(self) -> None:
      for i in range(self.N):
        chromosome = self.init_individual()
        self.population[i] = chromosome

    def tournament(self, num_selected: int) -> Tuple:
      selected = self.population[np.random.choice(self.N, size=(num_selected,))]
      dad = selected[0]
      mom = selected[1]
      it = 2
      while it < len(selected):
        if mom.fitness < selected[it].fitness:
          mom = selected[it]
        it += 1
      return dad, mom

    def selection(self) -> Tuple:
      dad, mom = self.tournament(self.opponents + 1)
      return dad, mom

    def split(self, chromosome: Solution) -> Tuple:
      ones = np.where(chromosome.cells == 1)[0]
      size = ones.size
      head = ones[:size//2]
      tail = ones[size//2:]
      return head, tail

    def feasible(self, head: np.ndarray, tail: np.ndarray) -> Solution:
      cells = np.zeros(self.problem.size)
      cells[head] = 1
      weight = np.sum(self.problem.weights[head])
      tail = np.setdiff1d(tail, head, assume_unique=False)
      np.random.shuffle(tail)
      for idx in tail:
        if weight + self.problem.weights[idx] <= self.problem.capacity:
          cells[idx] = 1
          weight += self.problem.weights[idx]
      fitness = self.problem.evaluate(cells)
      child = Solution(cells=cells, fitness=fitness, weight=weight,)
      return child

    def complete(self, chromosome: Solution):
      zeros = np.where(chromosome.cells == 0 )[0]
      np.random.shuffle(zeros)
      for idx in zeros:
        if chromosome.weight + self.problem.weights[idx] <= self.problem.capacity:
          chromosome.cells[idx] = 1
          chromosome.weight += self.problem.weights[idx]
          chromosome.fitness += self.problem.profits[idx]
      return chromosome


    def cross(self, dad: Solution, mom: Solution) -> Tuple:
      head_dad, tail_dad = self.split(dad)
      head_mom, tail_mom = self.split(mom)
      first_child = self.feasible(head_dad, tail_mom)
      first_child = self.complete(first_child)
      second_child = self.feasible(head_mom, tail_dad)
      second_child = self.complete(second_child)
      return (first_child, second_child)

    def mutation(self, chromosome: Solution) -> Solution:
      mut = np.random.uniform()
      if mut < 0.05:
        density = self.problem.profits / self.problem.weights
        order = np.argsort(density) # Menor a mayor densidad
        for idx in order:
          if chromosome.cells[idx] == 1:
            chromosome.cells[idx] = 0
            chromosome.weight -= self.problem.weights[idx]
            chromosome.fitness -= self.problem.profits[idx]
            break

        order = order[::-1] # Mayor a menor densidad
        best = []
        it = 0
        while len(best) <= 3 and it < order.size:
          idx = order[it]
          if (chromosome.cells[idx] == 0 and
              chromosome.weight + self.problem.weights[idx] <= self.problem.capacity):
            best.append(idx)
          it += 1
        added = np.random.choice(best)
        chromosome.cells[added] = 1
        chromosome.weight += self.problem.weights[added]
        chromosome.fitness += self.problem.profits[added]
      return chromosome

    def replace(self, population: np.ndarray) -> None:
      all = np.concatenate((self.population, population), dtype=object)
      self.population = np.array(sorted(all, key=lambda x: x.fitness, reverse=True)[:self.N], dtype=object)

    def solve(self) -> Solution:
      self.init_population()
      generation = 1
      while generation < self.generations:
        population = np.empty(shape=self.N, dtype=object)
        for i in range(0, self.N, 2):
          dad, mom = self.selection()
          first, second = self.cross(dad, mom)
          first = self.mutation(first)
          second = self.mutation(second)
          population[i] = first
          population[i+1] = second
        self.replace(population)
        generation += 1
      return self.population[0]
