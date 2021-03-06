#!/usr/bin/python3
from typing import Set, Tuple, Optional, Iterable

from PyQt5.QtCore import QLineF, QPointF

import time
import numpy as np
from TSPClasses import *
import heapq
from collections import defaultdict
import itertools
from queue import PriorityQueue


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario: Scenario):
        self._scenario = scenario

    """ 
    <summary>
    This is the entry point for the default solver
    which just finds a valid random tour.  Note this could be used to find your
    initial BSSF.
    </summary>
    <returns>
    results dictionary for GUI that contains three ints: cost of solution, 
    time spent to find solution, number of permutations tried during search, the 
    solution found, and three null values for fields not used for this 
    algorithm
    </returns> 
    """

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.cities
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    """
    <summary>
    This is the entry point for the greedy solver, which you must implement for 
    the group project (but it is probably a good idea to just do it for the branch-and
    bound project as a way to get your feet wet).  Note this could be used to find your
    initial BSSF.
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number of solutions found, the best
    solution found, and three null values for fields not used for this 
    algorithm</returns> 
    """

    def greedy(self, time_allowance=60.0):
        """
        Greedily find a valid TSP route.
        :param time_allowance: deprecated
        :return: a dictionary containing the route and information about how it was found
        """
        num_cities = len(self._scenario.cities)
        start_time = time.time()
        cities = self._scenario.cities  # save a reference for speedy look up
        count = 0

        route: List[City] = []
        for starting_city in cities:
            visited_cities = {starting_city}
            route = [starting_city]
            count += 1

            while len(visited_cities) < num_cities:
                next_city = self.find_closest_neighbor(route[-1], visited_cities)
                # if there is no valid neighbor to visit, break out of the loop
                if next_city is None:
                    break
                route.append(next_city)
                visited_cities.add(next_city)

            # check to make sure all the cities got visited and that  the starting city
            # is reachable from the end city
            if len(route) == num_cities and route[-1].cost_to(starting_city) < np.inf:
                # route.append(starting_city)
                break

        solution = TSPSolution(route)
        if solution.cost == np.inf:
            return self.defaultRandomTour(time_allowance)
        end_time = time.time()
        results = {'cost': solution.cost if len(route) == num_cities else np.inf,
                   'time': end_time - start_time,
                   'count': count,
                   'soln': solution,
                   'max': None,
                   'total': None,
                   'pruned': None}
        return results

    def find_closest_neighbor(self, city: City, visited_cities: Set[City]) -> Optional[City]:
        neighbors = ((city.cost_to(neighbor), neighbor)
                     for neighbor in self._scenario.cities
                     if neighbor not in visited_cities)
        try:
            return min(neighbors, key=lambda e: e[0])[1]
        except ValueError:
            return None

    """
    <summary>
    This is the entry point for the branch-and-bound algorithm that you will implement
    </summary>
    <returns>
    results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number solutions found during search (does
    not include the initial BSSF), the best solution found, and three more ints: 
    max queue size, total number of states created, and number of pruned states.
    </returns> 
    """

    def branchAndBound(self, time_allowance=60.0):
        pass

    """
        Ant colony algorithm
        * Start at city index 0
        * while node has children
        *    Calculate the priority of each path based on pheromone count
        *    Choose a random edge to follow (higher priority has more probability of being chosen)
        *    Once a valid solution is found, increment the pheromone count on each path in the solution.
        *        Increment size is based on the total cost of the solution
        *    Decrement all pheromone counts for all edges in the list of cities
        * End condition run ants in batches of BATCH_SIZE, and save the most common route as BSSF if better 
        * than last BSSF. Once PERCENT_ANTS are on same path return that result. 
    """

    def fancy(self, time_allowance=60.0):
        cities = self._scenario.cities
        ncities = len(cities)
        metThreshold = False
        count = 0

        threshold = .80  # the percent of cities that follow same route for route to be accepted
        batchSize = 50  # number of solutions per batch

        start_time = time.time()

        bssf = self.greedy(time_allowance)['soln']
        global RANDOM_COST
        RANDOM_COST = bssf.cost
        pheromoneMatrix = getPheromoneMatrix(ncities)

        convergences = []

        while not metThreshold and time.time() - start_time < time_allowance:
            # run a batch of ants and find solution

            batchRoutes = defaultdict(lambda: 0)
            numFound = 0

            while numFound < batchSize and time.time() - start_time < time_allowance:
                # runs an ant through the maze getting route then appending route to batchRoutes
                # route = [random.randrange(0, ncities)]  # Start at a random city index
                route = [0]
                costMatrix = getCostMatrix(cities)  # resets cost matrix each ant
                updateVisited(costMatrix, route[-1])  # set so cost matrix has infs for route
                antSuccess = True
                for i in range(ncities - 1):
                    # make the route
                    destinationIndex = getRandomEdge(costMatrix, pheromoneMatrix, route[-1])
                    if destinationIndex == -1:
                        antSuccess = False  # ends ant lifespan if reaches dead end
                        break

                    route.append(destinationIndex)

                    updateVisited(costMatrix, route[-1])  # set so cost matrix has infs for route
                if not antSuccess:
                    continue

                solverRoute = [cities[route[i]] for i in range(ncities)]

                thisSolution = TSPSolution(solverRoute)
                if thisSolution.cost != math.inf:
                    numFound += 1
                    batchRoutes[thisSolution] += 1
                    # increment pheromones
                    incrementPheromoneMatrix(pheromoneMatrix, route, thisSolution.cost)
                    if thisSolution.cost < bssf.cost:
                        bssf = thisSolution
                        # print('TIME:', time.time() - start_time, 'BSSF:', bssf.cost)
                        count += 1

                decrementMatrix(pheromoneMatrix)

            # If there are no valid solutions in the batch, don't calculate threshold
            if numFound == 0:
                break

            # If a single solution appears more than the defined threshold, return that solution
            maxNumSameSolution = max(batchRoutes.values())
            if maxNumSameSolution >= threshold * batchSize:
                metThreshold = True
            convergences.append(maxNumSameSolution / batchSize)

        end_time = time.time()
        print(convergences)
        global EXPECTED_EDGES, UNEXPECTED_EDGES
        print('unexpected', UNEXPECTED_EDGES, 'expected', EXPECTED_EDGES)
        results = {
            'cost': bssf.cost,
            'time': end_time - start_time,
            'count': count,
            'soln': bssf,
            'max': None,
            'total': None,
            'pruned': None
        }
        return results


# Returns a 2D Numpy Array (Adjacency matrix).
def getCostMatrix(cities):
    """
    O(n^2)
    """
    matrix = np.empty(shape=(len(cities), len(cities)))
    for i in range(len(cities)):
        city: City = cities[i]
        for j in range(len(cities)):
            matrix[i][j] = city.cost_to(cities[j])
    return matrix


# Returns 2D Numpy Array
def getPheromoneMatrix(numCities):
    """
    O(n^2)
    """
    return np.zeros(shape=(numCities, numCities))


def decrementMatrix(matrix: np.ndarray) -> None:
    """
    O(n^2)
    Used for decrementing all pheromone counts. Mutates matrix in place.
    :param matrix: a pheromone matrix
    :return: None
    """
    matrix *= .8


EXPECTED_EDGES = 0
UNEXPECTED_EDGES = 0


def getRandomEdge(costMatrix, pheromoneMatrix, parentCityIndex) -> int:
    """
    Calculates random edge while prioritizing higher pheromone counts
    :param costMatrix: the adjacency matrix of cities
    :param pheromoneMatrix: a matrix where the (i, j)th entry is the pheromone level on the
    edge between cities i and j
    :param parentCityIndex: The index of where the ant currently is
    :return: index of destination city
    """
    valid_cities = np.where(costMatrix[parentCityIndex] < np.inf)[0]
    # add 2 so that cities without pheromones have a chance of getting picked
    pheromone_level = pheromoneMatrix[parentCityIndex, valid_cities] + 1/RANDOM_COST
    probability_distribution = pheromone_level / sum(pheromone_level)
    edge = np.random.choice(valid_cities, p=probability_distribution) if len(valid_cities) > 0 else -1
    if edge > -1 and edge != max(zip(valid_cities, probability_distribution), key=lambda e: e[1])[0]:
        global UNEXPECTED_EDGES
        UNEXPECTED_EDGES += 1
    elif edge > -1:
        global EXPECTED_EDGES
        EXPECTED_EDGES += 1
    return edge


def updateVisited(costMatrix, destinationCity):
    """
    Make destinationCity unreachable. Modify costMatrix in place.
    """
    costMatrix[:, destinationCity] = np.inf


def incrementPheromoneMatrix(pheromoneMatrix: np.ndarray, route, cost):
    increaseVal = (5*len(route)) / cost
    for i in range(len(route) - 1):
        pheromoneMatrix[route[i]][route[i + 1]] += increaseVal
