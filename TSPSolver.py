#!/usr/bin/python3
from typing import Set, Tuple, Optional, Iterable

from PyQt5.QtCore import QLineF, QPointF

import time
import numpy as np
from TSPClasses import *
import heapq
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

        end_time = time.time()
        solution = TSPSolution(route)
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
    <summary>
    This is the entry point for the algorithm you'll write for your group project.
    </summary>
    <returns>
    results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number of solutions found during search, the 
    best solution found.  You may use the other three field however you like.
    algorithm
    </returns> 
    """

    def fancy(self, time_allowance=60.0):
        route = []
        """
        Ant colony algorithm
        * Start at city index 0
        * while node has children
        *    Calculate the priority of each path based on pheromone count
        *    Choose a random edge to follow (higher priority has more probability of being chosen)
        *    Once a valid solution is found, increment the pheromone count on each path in the solution.
        *        Increment size is based on the total cost of the solution
        *    Decrement all pheromone counts for all edges in the list of cities

        CLASSES:
            Edge
                - cost
                - pheromoneCount
            Matrix
                - Edge[][]
                - decrementAllPheromones()
                - randomizer here?
        """
        pass
