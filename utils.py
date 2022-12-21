from typing import Tuple, List


def read_file(filename: str) -> Tuple[int, int, List, List, float]:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    size, capacity = (int(x) for x in lines[0].split(" "))
    # remove the first line
    optimal = float(lines[-1].replace(",", "."))
    lines = lines[1:-1]

    # Read the weights and profits
    profits, weights = [], []
    for line in lines:
        line = line.replace(",", ".")
        profit, weight = (float(x) for x in line.split(" "))
        weights.append(weight)
        profits.append(profit)
    return size, capacity, weights, profits, optimal
