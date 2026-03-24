

import numpy as np
from numpy._typing import NDArray
import pytest

from genetic.tests.logger import CUDA_GA_TestLogger
from genetic.tests.constants import RANDOM_SEED



class PerceptronXOR:
    XOR_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    XOR_outputs = np.array([0, 1, 1, 0])
    
    def fitness(self, weights: NDArray[np.float64]) -> float:
        predictions = [self.activation_function(weights, x) for x in self.XOR_inputs]
        return np.sum(predictions == self.XOR_outputs)
    
    def activation_function(self, weights: NDArray[np.float64], inputs: NDArray[np.float64]) -> int:
        return 1 if np.dot(weights, inputs) >= 0 else 0
    
    
class GA_PerceptronXOR:
    def __init__(self, logger: CUDA_GA_TestLogger, population_size: int, generations: int, mutation_rate: float, gene_length: int):
        self.logger = logger.logger
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.gene_length = gene_length
        self.perceptron = PerceptronXOR()
        
    def _create_initial_population(self) -> NDArray[np.float64]:
        return np.random.uniform(-1, 1, (self.population_size, 2))
    
    def run(self):
        np.random.seed(RANDOM_SEED)
        population = self._create_initial_population()
        for generation in range(self.generations):
            # population individual is a weights vector
            fitnesses = np.array([self.perceptron.fitness(weights) for weights in population]) 
            self.logger.info(f"Calculated fitnesses. Sample: {fitnesses[:3]}")
            
            # top 50%
            sorted_indices = np.argsort(fitnesses)[::-1]
            population = population[sorted_indices]
            population = population[:self.population_size // 2]

            # crossover, create offspring for each parent pair
            offspring = []
            while len(offspring) < self.population_size // 2:
                parents = np.random.choice(len(population), 2, replace=False)
                child = self._crossover(population[parents[0]], population[parents[1]])
                
                # mutate
                child = self._mutate(child)
                offspring.append(child)
                
            population = np.vstack((population, offspring))
            best_fitness = fitnesses[sorted_indices[0]]
            self.logger.info(f"Generation {generation}, best fitness: {best_fitness}")

        best_weights = population[0]
        self.logger.info(f"Best weights: {best_weights}")
        return best_weights
        
    def _crossover(self, parent1: NDArray[np.float64], parent2: NDArray[np.float64]) -> NDArray[np.float64]:
        crossover_point = np.random.randint(1, self.gene_length)
        child = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        return child
        
    def _mutate(self, child: NDArray[np.float64]) -> NDArray[np.float64]:
        if np.random.rand() < self.mutation_rate:
            mut_idx = np.random.randint(self.gene_length)
            child[mut_idx] += np.random.normal()
        return child


TEST_PARAMS = [
    (10, 20),
    (50, 50),
    (100, 100),
    (500, 200),
    (1000, 1000),
    (1000, 2000),
]


@pytest.mark.parametrize("population_size, generations", TEST_PARAMS)
def test_perceptron_xor_ga(population_size, generations, logger: CUDA_GA_TestLogger):
    # Arrange
    test_logger = logger.logger
    test_logger.info("\n" + "=" * 80)
    test_logger.info("NEW TEST RUN")
    test_logger.info(f"Population size: {population_size}")
    test_logger.info(f"Generations: {generations}")

    mutation_rate = 0.1
    gene_length = 2  # XOR has 2 inputs

    params = {
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "gene_length": gene_length,
    }

    logger.start_test(
        f"test_xor_ga_pop_{population_size}_gens_{generations}"
    )
    logger.log_params(params)

    ga = GA_PerceptronXOR(
        logger=logger,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        gene_length=gene_length,
    )

    # Act
    best_weights = ga.run()

    # Assert
    test_logger.info("Testing perceptron with best weights.")
    
    perceptron = ga.perceptron
    predictions = [
        perceptron.activation_function(best_weights, x)
        for x in perceptron.XOR_inputs
    ]

    test_logger.info(f"Predictions: {predictions}")
    test_logger.info(f"Expected: {perceptron.XOR_outputs.tolist()}")
    final_test_fitness = sum(int(p == t) for p, t in zip(predictions, perceptron.XOR_outputs))
    
    passed = final_test_fitness >= 2
    assert passed, f"Perceptron cannot fully solve XOR; expected fitness >=2, got {final_test_fitness}"

    test_logger.info("XOR test passed with best weights.")
    test_logger.info("GA run completed. Test passed.")
    logger.end_test("passed" if passed else "failed", extra_info={"best_weights": best_weights})
