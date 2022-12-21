import numpy as np
from gwo import GWO
from ghs import GHS
from pso import PSO
from ga import GA

from knapsack import Problem, Solution
from pso import Solution as SolutionPSO
from utils import read_file


def main():
    np.random.seed(42)
    solutions = {}
    name = "Knapsack6"
    N, max_iterations = 20, 1000
    size, capacity, weights, profits, optimal = read_file(f"knapsack/data/{name}.txt")
    problem: Problem = Problem(
        name=name,
        size=size,
        capacity=capacity,
        weights=np.array(weights),
        profits=np.array(profits),
        optimal=optimal,
    )

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

    best = gwo.solve()
    solutions["GWO"] = best

    phi_p, phi_g, v_max = 0.8, 0.9, 0.9
    #phi_p, phi_g, v_max = 1, 1, 1
    omega = 1
    pso: PSO = PSO(
        problem=problem,
        max_iterations=max_iterations,
        N=N,
        population=np.empty(shape=N, dtype=object),
        omega=omega,
        phi_p=phi_p,
        phi_g=phi_g,
        v_max=v_max,
        global_best=SolutionPSO(np.zeros(size), 0, 0, np.zeros(size)),
    )
    
    best = pso.solve()
    solutions["PSO"] = best

    ga = GA(
          N=20,
          generations=1000,
          problem=problem,
          population=np.empty(shape=20, dtype=object),
          opponents=2,
      )
    best = ga.solve()
    solutions["GA"] = best

    N, HMCR, PAR = 20, 0.9, 0.3
    ghs: GHS = GHS(
        problem=problem,
        max_iterations=max_iterations,
        memory=np.empty(N, dtype=object),
        N=N,
        HMCR=HMCR,
        PAR=PAR,
    )
    
    best = ghs.solve()
    solutions["GHS"] = best
    for name, best in solutions.items():
        print(f"{name}")
        print(f'Fitness: {best.fitness} - {problem.evaluate(best.cells)}')
        print(f'Weigh: {best.weight} - {problem.weigh(best.cells)}')
        print(f'Capacity: {problem.capacity} - Optimal: {problem.optimal}')
        print(f'{best.cells}')

if __name__ == "__main__":
    main()
