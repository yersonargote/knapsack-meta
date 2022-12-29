import numpy as np

from ga import GA
from ghs import GHS
from gwo import GWO
from knapsack import Problem, Solution
from pso import PSO
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
    best = gwo.solve()
    solutions["GWO"] = best

    # PSO
    phi_p, phi_g, v_max = 0.8, 0.9, 0.9
    # phi_p, phi_g, v_max = 1, 1, 1
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

    # GA
    ga = GA(
        N=N,
        generations=max_iterations,
        problem=problem,
        population=np.empty(shape=N, dtype=object),
        opponents=2,
    )
    best = ga.solve()
    solutions["GA"] = best

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

    best = ghs.solve()
    solutions["GHS"] = best

    for name, best in solutions.items():
        print(f"{name}")
        print(f"Capacity: {problem.capacity} - Optimal: {problem.optimal}")
        valid = np.isclose(problem.evaluate(best.cells), best.fitness)
        print(f"Fitness: {best.fitness} - Valid: {valid}")
        valid = np.isclose(best.weight, problem.weigh(best.cells))
        print(f"Weigh: {best.weight} - Valid {valid}")
        print(f"{best.cells}")


if __name__ == "__main__":
    main()
