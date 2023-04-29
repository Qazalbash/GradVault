from abc import ABC, abstractmethod


# Abstract class for any optimization problem
class Problem(ABC):
    """Abstract class for any optimization problem

    Args:
        ABC (): Abstract Base Class
    """

    inverse_fitness: bool = False

    @abstractmethod
    def chromosome(self) -> list:
        """Return a random chromosome

        Returns:
            list: Random chromosome
        """
        pass

    @abstractmethod
    def fitness_function(self) -> float:
        """Return the fitness of a chromosome

        Returns:
            float: Fitness of a chromosome
        """
        pass

    @abstractmethod
    def mutate(self) -> list:
        """Return a mutated chromosome

        Returns:
            list: Mutated chromosome
        """
        pass

    @abstractmethod
    def crossover(self):
        """Return a crossovered chromosome
        """
        pass
