import time

import numpy as np
import typer
from rich import print

from ga import GA
from ghs import GHS
from gwo import GWO
from knapsack import Problem, Solution
from pso import PSO
from pso import Solution as SolutionPSO
from utils import read_file


def main(name: str = typer.Argument("Knapsack6")):
    global time
    np.random.seed(42)
    solutions = {}
    N, max_iterations, v_max = 20, 1000, 1
    size, capacity, weights, profits, optimal = read_file(f"knapsack/data/{name}.txt")
    problem: Problem = Problem(
        name=name,
        size=size,
        capacity=capacity,
        weights=np.array(weights),
        profits=np.array(profits),
        optimal=optimal,
    )

    # GWO
    gwo: GWO = GWO(
        max_iterations=max_iterations,
        N=N,
        problem=problem,
        population=np.empty(shape=N, dtype=object),
        a=0,
        alpha=Solution(np.zeros(size), 0, 0),
        beta=Solution(np.zeros(size), 0, 0),
        delta=Solution(np.zeros(size), 0, 0),
    )
    start = time.time()
    best = gwo.solve()
    end = time.time()
    solutions["GWO"] = (best, end - start)

    # PSO
    alpha, beta, delta, epsilon = 0.9, 1.0, 1.0, 0.5
    pso: PSO = PSO(
        problem=problem,
        max_iterations=max_iterations,
        N=N,
        population=np.empty(shape=N, dtype=object),
        best=SolutionPSO(np.zeros(size), 0, 0, np.zeros(size)),
        alpha=alpha,
        beta=beta,
        delta=delta,
        epsilon=epsilon,
        v_max=v_max,
    )
    start = time.time()
    best = pso.solve()
    end = time.time()
    solutions["PSO"] = (best, end - start)

    # GA
    ga = GA(
        N=N,
        generations=max_iterations,
        problem=problem,
        population=np.empty(shape=N, dtype=object),
        opponents=2,
    )
    start = time.time()
    best = ga.solve()
    end = time.time()
    solutions["GA"] = (best, end - start)

    # GHS
    N, HMCR, PAR = 20, 0.9, 0.3
    ghs: GHS = GHS(
        problem=problem,
        max_iterations=max_iterations,
        memory=np.empty(N, dtype=object),
        N=N,
        HMCR=HMCR,
        PAR=PAR,
    )
    start = time.time()
    best = ghs.solve()
    end = time.time()
    solutions["GHS"] = (best, end - start)

    for name, (best, time) in solutions.items():
        print(f"{name} - Time: {time}")
        print(f"Optimal: {problem.optimal} - Capacity: {problem.capacity}")
        valid = np.isclose(problem.evaluate(best.cells), best.fitness)
        print(f"Fitness: {best.fitness} - Valid: {valid}")
        valid = np.isclose(best.weight, problem.weigh(best.cells))
        print(f"Weigh: {best.weight} - Valid: {valid}")
        print(f"{best.cells}")


if __name__ == "__main__":
    typer.run(main)
