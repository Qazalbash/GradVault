import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import random

from Evolution.problem import Problem
from Knapsack.data import knapsack_capacity as threshold
from Knapsack.data import number_of_items, profits, weights


class Knapsack(Problem):

    @staticmethod
    def chromosome() -> list:
        """Returns a list of randomized binary numbers

        Returns:
            list: list of randomized binary numbers
        """
        return random.choices([0, 1], k=number_of_items)

    @staticmethod
    def fitness_function(solution: list) -> int:
        """Calculates the fitness of a solution

        Args:
            solution (list): list of binary numbers

        Returns:
            int: fitness of a solution (total profit)

        Description:
            We loop over the solution. If the solution has a 1 in the ith 
            position, we add the profit of the ith item to the total profit. 
            However, if we exceeed the capacity of the knapsack, the fitness
            is 0.
        """
        total_profit, total_weight = 0, 0
        for binary, profit, weight in zip(solution, profits, weights):
            if binary == 1:
                total_profit += profit
                total_weight += weight
        return total_profit * (total_weight <= threshold)

    @staticmethod
    def mutate(individual: list) -> list:
        """Mutates the individual by flipping a bit

        Args:
            individual (list): list of binary numbers

        Returns:
            list: list of binary numbers after mutation
        """
        index = random.randint(0, len(individual) - 1)
        individual[index] = int(not individual[index])
        return individual

    @staticmethod
    def crossover(parent1: list, parent2: list) -> list:
        """Returns a child after breeding from two parents

        Args:
            parent1 (list): first parent
            parent2 (list): second parent

        Returns:
            list: child after breeding from two parents
        """
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        startGene, endGene = min(geneA, geneB), max(geneA, geneB)
        childP1 = parent1[startGene:endGene]
        childP2 = [gene for gene in parent2 if gene not in childP1]
        return childP1 + childP2
