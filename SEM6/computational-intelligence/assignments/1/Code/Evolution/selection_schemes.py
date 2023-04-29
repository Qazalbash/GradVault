import random


class SelectionSchemes:

    def __init__(self, fitness_function, population_size: int) -> None:
        """Initializes the selection schemes class with the fitness function
        and population size

        Args:
            fitness_function (function): fitness function
            population_size (int): population size

        Description:
            Population size is the number of chromosones to select from the 
            population. fitness_function is the fitness function to calculate
            fitness.
        """
        self.population_size = population_size
        self.fitness_function = fitness_function

    def truncation(self, population: list, T: float = 0.4) -> list:
        """Truncates the population according to the truncation value

        Args:
            population (list): population of chromosomes
            T (float, optional): truncation value. Defaults to 0.4.

        Returns:
            list: new population of chromosomes after truncation

        Description:
            First the population will be sorted according to the fitness of 
            each chromosome. Then it will be truncated according to the 
            truncation value (T). The truncation value is a value between 0 and
            1. T = 0.4 means that we will consider 60% of the population
            (fittest) and select random chromosomes from that 60% and make a
            new population.
        """
        N = len(population)
        sorted_population = sorted(population, key=self.fitness_function)
        new_population = list(
            map(
                lambda _: sorted_population[random.randint(
                    int((1 - T) * N), N - 1)], range(self.population_size)))
        return new_population

    def fitness_proportionate(self, population: list) -> list:
        """Selects chromosomes according to their fitness value (probability)

        Args:
            population (list): population of chromosomes

        Returns:
            list: new population of chromosomes after selection

        Description:
            Each chromosome will be assigned a probability of being selected.
            Such that the probability of a chromosome being selected is
            (fitness of chromosone) / (sum of all fitnesses).
        """
        fitness_lst = list(map(self.fitness_function, population))
        total_fitness = sum(fitness_lst)
        probabilities = list(map(lambda x: x / total_fitness, fitness_lst))
        return random.choices(population, probabilities, k=self.population_size)

    def tournament_selection(self, population: list) -> list:
        """Selects chromosomes according to tournament selection

        Args:
            population (list): population of chromosomes

        Returns:
            list: new population of chromosomes after selection

        Description:
            We will select two random chromosomes from the population and
            select the chromosome with the highest fitness and add that 
            chromosone to the new population.
        """
        new_population = []
        for _ in range(self.population_size):
            tournament = random.sample(population, 2)
            winner = max(tournament, key=self.fitness_function)
            new_population.append(winner)
        return new_population

    def ranked_selection(self, population: list) -> list:
        """Selects chromosomes according to their rank

        Args:
            population (list): population of chromosomes

        Returns:
            list: new population of chromosomes after selection

        Description:
            The population will be sorted according to the fitness of each
            chromosome. The probability of a chromosome being selected is (rank
            of chromosome) / (total number of chromosomes). The fittest
            invidual will have the highest rank which will be N. The least fit
            will have the lowest rank which will be 1.
        """
        N = len(population)
        sorted_population = sorted(population, key=self.fitness_function)
        probabilities = list(map(lambda x: x / N, range(1, N + 1)))
        return random.choices(sorted_population,
                              probabilities,
                              k=self.population_size)

    def random_selection(self, population: list) -> list:
        """Selects chromosomes randomly from the population

        Args:
            population (list): population of chromosomes

        Returns:
            list: new population of chromosomes after selection
        """
        return random.choices(population, k=self.population_size)
