import unittest
from TSPSolver import getPheromoneMatrix, decrementMatrix, incrementPheromoneMatrix


def printMatrix(matrix):
    text = ''
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            text += str(matrix[i][j]) + '   '
        text += '\n'
    text += '\n'
    print(text)


numCities = 8
pheromoneMatrix = getPheromoneMatrix(numCities)
printMatrix(pheromoneMatrix)

route = [0, 3, 1, 4, 2]
cost = 9
incrementPheromoneMatrix(pheromoneMatrix, route, cost)
printMatrix(pheromoneMatrix)

decrementMatrix(pheromoneMatrix)
printMatrix(pheromoneMatrix)
