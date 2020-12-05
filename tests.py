import unittest
from TSPSolver import getPheromoneMatrix, decrementMatrix, incrementPheromoneMatrix
import numpy as np

print('Original Matrix')
numCities = 8
pheromoneMatrix = getPheromoneMatrix(numCities)
print(pheromoneMatrix)
print()

print('Incremented Matrix')
route = [0, 3, 1, 4, 2]
cost = 500
incrementPheromoneMatrix(pheromoneMatrix, route, cost)
print(pheromoneMatrix)
print()

decrementMatrix(pheromoneMatrix)
print('Decremented Matrix')
print(pheromoneMatrix)
print()


class TestMatrixOperations(unittest.TestCase):
    def test_update_visited(self):
        adj_matrix = self.make_fresh_adj_matrix()

    def test_get_random_edge(self):
        cost_matrix = self.make_fresh_adj_matrix()
        # assume the pheromone levels are the same as adjacency matrix
        pheromone_matrix = self.make_fresh_adj_matrix()

    @staticmethod
    def make_fresh_adj_matrix() -> np.ndarray:
        adj_matrix = np.array([
            [0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0]
        ])
        return adj_matrix
