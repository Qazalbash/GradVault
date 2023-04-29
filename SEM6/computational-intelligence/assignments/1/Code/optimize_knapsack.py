from Knapsack.knapsack import Knapsack
from optimization import Optimization
from selection_cases import selection_cases

TOTAL_CASES = 10


def list_of_problems(problem, pop_sizes: list, num_offsprings: int,
                     num_iterations: int, num_gens: int, mutation_rates: float):
    optimizations = []

    for i in range(TOTAL_CASES):
        case = selection_cases[i]
        optimizations.append(
            Optimization(problem=problem,
                         selection_case=case,
                         population_size=pop_sizes[i],
                         mutation_rate=mutation_rates[i],
                         number_of_generations=num_gens[i],
                         number_of_offsprings=num_offsprings[i],
                         number_of_iterations=num_iterations[i]))

    return optimizations


problem = Knapsack
pop_sizes = [150 for _ in range(TOTAL_CASES)]
num_offsprings = [70 for _ in range(TOTAL_CASES)]
num_iterations = [10 for _ in range(TOTAL_CASES)]
num_gens = [1000 for _ in range(TOTAL_CASES)]
mutation_rates = [0.45 for _ in range(TOTAL_CASES)]

optimization_TSP_BSF = list_of_problems(problem,
                                        pop_sizes=pop_sizes,
                                        num_offsprings=num_offsprings,
                                        num_iterations=num_iterations,
                                        num_gens=num_gens,
                                        mutation_rates=mutation_rates)

pop_sizes = [150 for _ in range(TOTAL_CASES)]
num_offsprings = [70 for _ in range(TOTAL_CASES)]
num_iterations = [20 for _ in range(TOTAL_CASES)]
num_gens = [1000 for _ in range(TOTAL_CASES)]
mutation_rates = [0.45 for _ in range(TOTAL_CASES)]

optimization_TSP_ASF = list_of_problems(problem,
                                        pop_sizes=pop_sizes,
                                        num_offsprings=num_offsprings,
                                        num_iterations=num_iterations,
                                        num_gens=num_gens,
                                        mutation_rates=mutation_rates)

for i in range(TOTAL_CASES):
    op_b: Optimization = optimization_TSP_BSF[i]
    op_b.plot_BSF()
    op_a: Optimization = optimization_TSP_ASF[i]
    op_a.plot_ASF()
