import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import random

from Evolution.problem import Problem
from Graph_Coloring.data import is_valid, num_nodes


class Graph_Coloring(Problem):
    """Graph Coloring Problem

    Args:
        Problem (class): Problem class from Evolution.problem
    """

    inverse_fitness = True

    @staticmethod
    def chromosome() -> list:
        """Returns a list of random colors

        Returns:
            list: list of random colors

        Description:
            A random solution would be assign a different color to each vertex
        """
        return random.sample(range(num_nodes), num_nodes)

    @staticmethod
    def fitness_function(individual: list) -> float:
        """Calculates the fitness of a solution

        Args:
            individual (list): list of colors

        Returns:
            float: fitness of a solution (number of colors used)

        Description:
            If the solution is valid, return the number of colors used. else,
            fitness is zero.
        """
        if is_valid(individual):
            return 1 / len(set(individual))
        return 0.0

    @staticmethod
    def crossover(parent1: list, parent2: list) -> list:
        """Returns a child solution by crossing over two parents

        Args:
            parent1 (list): first parent
            parent2 (list): second parent

        Returns:
            list: child solution
        """
        position = random.randint(0, num_nodes - 1)
        child = parent1[:position] + parent2[position:]
        return child

    @staticmethod
    def mutate(individual: list) -> list:
        """Mutates the individual by changing a color

        Args:
            individual (list): list of colors

        Returns:
            list: list of colors after mutation
        """
        position = random.randint(0, num_nodes - 1)
        individual[position] = random.randint(0, num_nodes - 1)
        return individual
