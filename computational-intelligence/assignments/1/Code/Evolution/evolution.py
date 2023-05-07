import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import random

from Evolution.problem import Problem
from Evolution.selection_schemes import SelectionSchemes


class Evolution:

    def __init__(self,
                 problem: Problem,
                 selection_method1: int = 0,
                 selection_method2: int = 0,
                 population_size: int = 30,
                 number_of_offsprings: int = 10,
                 number_of_generations: int = 100,
                 mutation_rate: float = 0.50) -> None:

        self.selection_schemes = SelectionSchemes(
            population_size=population_size,
            fitness_function=problem.fitness_function)

        self.chromosome = problem.chromosome
        self.fitness_function = problem.fitness_function
        self.mutate = problem.mutate
        self.crossover = problem.crossover

        self.selection_method1 = selection_method1
        self.selection_method2 = selection_method2

        self.population_size = population_size
        self.number_of_offsprings = number_of_offsprings
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate

    def initial_population(self) -> list:
        """Initial population of chromosomes

        Returns:
            list: List of chromosomes of length population_size
        """
        return [self.chromosome() for _ in range(self.population_size)]

    def get_best_individual(self, population: list):
        """Get the fittest chromosome from the population

        Args:
            population (list): List of chromosomes
        """
        return max(population, key=self.fitness_function)

    def get_best_fitness(self, population: list) -> float:
        """Get the best fitness of the population

        Args:
            population (list): List of chromosomes

        Returns:
            float: Best fitness of the population
        """
        best_chromosome = self.get_best_individual(population)
        return self.fitness_function(best_chromosome)

    def truncation(self, population: list) -> list:
        """Truncation selection method

        Args:
            population (list): List of chromosomes

        Returns:
            list: List of chromosomes after truncation
        """
        return self.selection_schemes.truncation(population)

    def fitness_proportionate(self, population: list) -> list:
        """Fitness Propotionate Selection method

        Args:
            population (list): List of chromosomes

        Returns:
            list: List of chromosomes after fitness proportionate selection
        """
        return self.selection_schemes.fitness_proportionate(population)

    def tournament_selection(self, population: list) -> list:
        """Binary Tournament selection method

        Args:
            population (list): List of chromosomes

        Returns:
            list: List of chromosomes after tournament selection
        """
        return self.selection_schemes.tournament_selection(population)

    def ranked_selection(self, population: list) -> list:
        """Ranked based selection method

        Args:
            population (list): List of chromosomes

        Returns:
            list: List of chromosomes after ranked based selection
        """
        return self.selection_schemes.ranked_selection(population)

    def random_selection(self, population: list) -> list:
        """Random selection method

        Args:
            population (list): List of chromosomes

        Returns:
            list: List of chromosomes after random selection
        """
        return self.selection_schemes.random_selection(population)

    def get_offspring(self, parent1, parent2):
        """Get the offspring of two parents using crossover and mutation

        Args:
            parent1 (_type_): Parent 1
            parent2 (_type_): Parent 2

        Returns:
            _type_: Offspring of parent 1 and parent 2 after crossover and mutation
        """
        child = self.crossover(parent1, parent2)
        if random.random() < self.mutation_rate:
            return self.mutate(child)
        return child

    def breed_parents(self, population: list) -> list:
        """Breed the parents to get the offsprings

        Args:
            population (list): List of chromosomes

        Returns:
            list: List of chromosomes after breeding
        """
        for _ in range(self.number_of_offsprings):
            parents = random.sample(population, 2)
            child = self.get_offspring(parents[0], parents[1])
            population.append(child)
        return population

    def next_generation(self, population: list) -> list:
        """Get the next generation using selection methods and breeding of the parents

        Args:
            population (list): List of chromosomes

        Returns:
            list: List of chromosomes after selection and breeding
        """
        selection_methods = [
            self.fitness_proportionate, self.ranked_selection,
            self.tournament_selection, self.truncation, self.random_selection
        ]

        parents = selection_methods[self.selection_method1](population)
        new_population = self.breed_parents(parents)
        survivors = selection_methods[self.selection_method2](new_population)
        return survivors

    def step(self, population: list) -> tuple[list, float]:
        """Get population of generation and best fitness of 
        the population after selection and breeding of the parents

        Args:
            population (list): List of chromosomes

        Returns:
            tuple[list, float]: List of chromosomes after 
            selection and breeding and best fitness of the population
        """
        return self.next_generation(population), self.get_best_fitness(
            population)

    def run(self) -> list:
        """Run the evolution

        Returns:
            list: List of best fitness of the population
        """
        population = self.initial_population()
        fitness_lst = []
        for _ in range(self.number_of_generations):
            population, best_fitness = self.step(population)
            fitness_lst.append(best_fitness)
        return fitness_lst
