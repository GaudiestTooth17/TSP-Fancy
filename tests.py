import unittest
from TSPSolver import getPheromoneMatrix, decrementMatrix, incrementPheromoneMatrix

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
