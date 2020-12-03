import random
import unittest
from typing import List

from PyQt5.QtCore import QPointF
import numpy as np

from TSPClasses import Scenario
import TSPSolver


class TestFunctionsOnMatrices(unittest.TestCase):
    def test_reduce_matrix(self):
        # this matrix comes from slide 30 of lecture 21
        adjacency_matrix = np.array([[np.inf, 9, np.inf, 8, np.inf],
                                     [np.inf, np.inf, 4, np.inf, 2],
                                     [np.inf, 3, np.inf, 4, np.inf],
                                     [np.inf, 6, 7, np.inf, 12],
                                     [1, np.inf, np.inf, 10, np.inf]])

        # this comes from slide 31 of lecture 21
        expected_root_matrix = np.array([[np.inf, 1, np.inf, 0, np.inf],
                                         [np.inf, np.inf, 1, np.inf, 0],
                                         [np.inf, 0, np.inf, 1, np.inf],
                                         [np.inf, 0, 0, np.inf, 6],
                                         [0, np.inf, np.inf, 9, np.inf]])
        actual_root_matrix, root_cost = TSPSolver.reduce_matrix(adjacency_matrix, None)
        self.assertTrue((expected_root_matrix == actual_root_matrix).all(), 'The root matrix was reduced incorrectly.')
        self.assertEqual(21, root_cost, 'An incorrect lower bound was returned.')

        # this comes from the child going to state 2 on slide 34
        expected_child = np.array([[np.inf, np.inf, np.inf, np.inf, np.inf],
                                   [np.inf, np.inf, 1, np.inf, 0],
                                   [np.inf, np.inf, np.inf, 0, np.inf],
                                   [np.inf, np.inf, 0, np.inf, 6],
                                   [0, np.inf, np.inf, 9, np.inf]])
        actual_child, additional_cost = TSPSolver.reduce_matrix(expected_root_matrix, (City(1), City(0)))
        self.assertTrue((expected_child == actual_child).all(), f'The child matrix was reduced incorrectly\n{actual_child}')
        self.assertEqual(23, root_cost + additional_cost, "The child's lower bound was calculated incorrectly.")

    def test_make_children(self):
        pass


class City:
    """
    This is just a mock city used for testing purposes
    """
    def __init__(self, index: int):
        self.index = index
