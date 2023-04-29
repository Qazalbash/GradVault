import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import random

from Evolution.problem import Problem
from Travelling_Salesman.data import matrix as graph
from Travelling_Salesman.data import num_nodes as num_cities


class TSP(Problem):
    inverse_fitness = True

    @staticmethod
    def chromosome() -> list:
        """Returns a random route of cities

        Returns:
            list: A random route of cities
        """
        return random.sample(list(range(num_cities)), num_cities)

    @staticmethod
    def fitness_function(route: list) -> float:
        """Calculates the distance covered in the route

        Args:
            route (list): different routes of cities

        Returns:
            float: distance covered in the route
        """
        N = len(graph) - 1
        distances = list(map(lambda x: graph[route[x]][route[x + 1]], range(N)))
        return 1 / sum(distances)

    @staticmethod
    def mutate(individual: list) -> list:
        """Mutates the route by swapping two cities

        Args:
            individual (list): list of cities

        Returns:
            list: list of cities after mutation
        """
        indexes = random.sample(list(range(len(individual))), 2)
        swap1, swap2 = indexes[0], indexes[1]
        individual[swap1], individual[swap2] = individual[swap2], individual[
            swap1]
        return individual

    @staticmethod
    def crossover(parent1: list, parent2: list) -> list:
        """Returns a offspring after breeding from two parents

        Args:
            parent1 (list): first parent
            parent2 (list): second parent

        Returns:
            list: offspring after breeding from two parents
        """
        gene1 = int(random.random() * len(parent1))
        gene2 = int(random.random() * len(parent1))

        start_gene = min(gene1, gene2)
        end_gene = max(gene1, gene2)
        child1 = parent1[start_gene:end_gene]
        child2 = [gene for gene in parent2 if gene not in child1]
        child = child1 + child2
        return child
