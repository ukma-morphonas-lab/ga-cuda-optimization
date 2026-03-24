# pyright: reportCallIssue=false

# V2: shared memory usage

from time import time
from numba.core.types import logging
import numpy as np
from numpy._typing import NDArray
import pytest
from numba import cuda, float32, int32

from genetic.tests.logger import CUDA_GA_TestLogger
from genetic.tests.constants import RANDOM_SEED

numba_logger = logging.getLogger('numba.cuda.cudadrv.driver')
numba_logger.setLevel(logging.WARNING)

@cuda.jit
def fitness_kernel(population, inputs, outputs, fitnesses, pop_size, num_inputs):
    # caching XOR inputs/outputs in shared memory for faster reuse inside the block
    shared_inputs = cuda.shared.array(shape=(8,), dtype=float32)  # type: ignore[call-arg]  # 4 samples * 2 inputs
    shared_outputs = cuda.shared.array(shape=(4,), dtype=int32)  # type: ignore[call-arg]

    tx = cuda.threadIdx.x
    if tx < num_inputs * 2:
        shared_inputs[tx] = inputs[tx] # type: ignore
    if tx < num_inputs:
        shared_outputs[tx] = outputs[tx] # type: ignore
    cuda.syncthreads()

    idx = cuda.grid(1) # type: ignore
    
    if idx < pop_size:
        correct = 0
        weights_start = idx * 2  # each individual has 2 weights
        
        # among 4 XOR input combinations
        for i in range(num_inputs):
            # dot product: weights · inputs
            dot_product = 0.0
            for j in range(2):
                dot_product += population[weights_start + j] * shared_inputs[i * 2 + j] # type: ignore
            
            # activation function
            prediction = 1 if dot_product >= 0 else 0
            
            # check if prediction matches expected output
            if prediction == shared_outputs[i]:  # type: ignore[index]
                correct += 1
        
        fitnesses[idx] = correct


@cuda.jit
def crossover_kernel(parents, offspring, crossover_points, pop_size, gene_length):
    # each thread creates one offspring from two parents
    idx = cuda.grid(1) # type: ignore
    
    if idx < pop_size:
        parent1_idx = idx * 2 * gene_length
        parent2_idx = (idx * 2 + 1) * gene_length # type: ignore
        offspring_idx = idx * gene_length
        crossover_point = crossover_points[idx]
        
        # copy from parent1 up to crossover point
        for i in range(crossover_point):
            offspring[offspring_idx + i] = parents[parent1_idx + i]
        
        # copy from parent2 after crossover point
        for i in range(crossover_point, gene_length):
            offspring[offspring_idx + i] = parents[parent2_idx + i]


@cuda.jit
def mutation_kernel(population, mutation_values, mutation_flags, pop_size, gene_length):
    # each thread mutates one individual
    idx = cuda.grid(1) # type: ignore
    
    if idx < pop_size:
        if mutation_flags[idx] == 1:
            mut_idx = int(mutation_values[idx * 2 + 1]) # type: ignore
            # + random mutation value
            population[idx * gene_length + mut_idx] += mutation_values[idx * 2]


class PerceptronXOR:
    XOR_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    XOR_outputs = np.array([0, 1, 1, 0])
    
    def fitness(self, weights: NDArray[np.float64]) -> float:
        predictions = [self.activation_function(weights, x) for x in self.XOR_inputs]
        return np.sum(predictions == self.XOR_outputs)
    
    def activation_function(self, weights: NDArray[np.float64], inputs: NDArray[np.float64]) -> int:
        return 1 if np.dot(weights, inputs) >= 0 else 0


class GA_PerceptronXOR:
    def __init__(self, logger, population_size: int, generations: int, 
                 mutation_rate: float, gene_length: int, gpu_enabled: bool = False,
                 seed: int = RANDOM_SEED):
        self.logger = logger.logger
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.gene_length = gene_length
        self.gpu_enabled = gpu_enabled
        self.perceptron = PerceptronXOR()
        self.seed = seed
        
        self.fitness_compute_time = 0
        self.gpu_transfer_time = 0
        self.gpu_compute_time = 0
        self.total_fitness_calls = 0
        
        if self.gpu_enabled:
            self._prepare_gpu_data()
    
    def _prepare_gpu_data(self):
        transfer_start = time()
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # constant data
        self.XOR_inputs_flat = self.perceptron.XOR_inputs.flatten().astype(np.float32)
        self.XOR_outputs_flat = self.perceptron.XOR_outputs.astype(np.int32)
        
        # transfer
        self.d_inputs = cuda.to_device(self.XOR_inputs_flat)
        self.d_outputs = cuda.to_device(self.XOR_outputs_flat)
        
        self.gpu_transfer_time += time() - transfer_start
        self.logger.info("GPU preparation complete: XOR data transferred")
    
    def _create_initial_population(self) -> NDArray[np.float64]:
        return np.random.uniform(-1, 1, (self.population_size, self.gene_length))
    
    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        population = self._create_initial_population()
        
        for generation in range(self.generations):
            fitnesses = self._compute_fitness(population)

            # top 50%
            sorted_indices = np.argsort(fitnesses)[::-1]
            population = population[sorted_indices]
            elite = population[:self.population_size // 2].copy()

            # crossover and mutation
            if self.gpu_enabled:
                offspring = self._crossover_mutation_gpu(elite)
            else:
                offspring = self._crossover_mutation_cpu(elite)

            population = np.vstack((elite, offspring))
            best_fitness = fitnesses[sorted_indices[0]]

            # log every 100th or last 
            if generation % 100 == 0 or generation == self.generations - 1:
                self.logger.info(f"Generation {generation}, fitness sample: {fitnesses[:3]}")
                self.logger.info(f"Generation {generation}, best fitness: {best_fitness}, gpu_enabled: {self.gpu_enabled}")
            
            if best_fitness == 4:
                self.logger.info(f"Perfect solution found at generation {generation}!")
                break

        best_weights = population[0]
        self.logger.info(f"Best weights: {best_weights}")
        
        if self.gpu_enabled:
            self.logger.info(f"GPU transfer time: {self.gpu_transfer_time:.4f}s")
            self.logger.info(f"GPU compute time: {self.gpu_compute_time:.4f}s")
        self.logger.info(f"Total fitness compute time: {self.fitness_compute_time:.4f}s")
        
        return best_weights
    
    def _compute_fitness(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        start_time = time()
        self.total_fitness_calls += 1
        
        if self.gpu_enabled:
            fitnesses = self._compute_fitness_gpu(population)
        else:
            fitnesses = self._compute_fitness_cpu(population)
        
        self.fitness_compute_time += time() - start_time
        return fitnesses
    
    def _compute_fitness_cpu(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([self.perceptron.fitness(weights) for weights in population])
    
    def _compute_fitness_gpu(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        pop_size = len(population)
        num_inputs = 4  # XOR has 4 test cases
        
        transfer_start = time()
        
        # flatten population and transfer
        population_flat = population.flatten().astype(np.float64)
        d_population = cuda.to_device(population_flat)
        d_fitnesses = cuda.device_array(pop_size, dtype=np.float64)
        
        self.gpu_transfer_time += time() - transfer_start
        
        # launch kernel
        threads_per_block = 256
        blocks_per_grid = (pop_size + threads_per_block - 1) // threads_per_block
        
        compute_start = time()
        fitness_kernel[blocks_per_grid, threads_per_block]( # type: ignore
            d_population, self.d_inputs, self.d_outputs, 
            d_fitnesses, pop_size, num_inputs
        )
        cuda.synchronize()
        self.gpu_compute_time += time() - compute_start
        
        # transfer results back
        transfer_start = time()
        fitnesses = d_fitnesses.copy_to_host()
        self.gpu_transfer_time += time() - transfer_start
        
        return fitnesses.astype(np.float64)
    
    def _crossover_mutation_cpu(self, elite: NDArray[np.float64]) -> NDArray[np.float64]:
        offspring = []
        while len(offspring) < self.population_size // 2:
            parents = np.random.choice(len(elite), 2, replace=False)
            child = self._crossover(elite[parents[0]], elite[parents[1]])
            child = self._mutate(child)
            offspring.append(child)
        return np.array(offspring)
    
    def _crossover_mutation_gpu(self, elite: NDArray[np.float64]) -> NDArray[np.float64]:
        num_offspring = self.population_size // 2
        
        transfer_start = time()
        
        # prepare parent pairs for crossover
        parent_pairs = []
        for _ in range(num_offspring):
            parents = np.random.choice(len(elite), 2, replace=False)
            parent_pairs.extend([elite[parents[0]], elite[parents[1]]])
        parent_pairs_array = np.array(parent_pairs, dtype=np.float64).flatten()
        
        # random crossover points
        crossover_points = np.random.randint(1, self.gene_length, num_offspring, dtype=np.int32)
        
        # prepare mutation data
        mutation_flags = (np.random.rand(num_offspring) < self.mutation_rate).astype(np.int32)
        mutation_indices = np.random.randint(0, self.gene_length, num_offspring, dtype=np.int32)
        mutation_vals = np.random.normal(0, 1, num_offspring).astype(np.float64)
        
        # interleave mutation values and indices
        # https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8 
        mutation_data = np.zeros(num_offspring * 2, dtype=np.float64)
        mutation_data[::2] = mutation_vals
        mutation_data[1::2] = mutation_indices
        
        # transfer to GPU
        d_parents = cuda.to_device(parent_pairs_array)
        d_crossover_points = cuda.to_device(crossover_points)
        d_offspring = cuda.device_array(num_offspring * self.gene_length, dtype=np.float64)
        d_mutation_data = cuda.to_device(mutation_data)
        d_mutation_flags = cuda.to_device(mutation_flags)
        
        self.gpu_transfer_time += time() - transfer_start
        
        # crossover kernel
        threads_per_block = 256
        blocks_per_grid = (num_offspring + threads_per_block - 1) // threads_per_block
        
        compute_start = time()
        crossover_kernel[blocks_per_grid, threads_per_block]( # type: ignore
            d_parents, d_offspring, d_crossover_points, 
            num_offspring, self.gene_length
        )
        
        # mutation kernel
        mutation_kernel[blocks_per_grid, threads_per_block]( # type: ignore
            d_offspring, d_mutation_data, d_mutation_flags,
            num_offspring, self.gene_length
        )
        
        cuda.synchronize()
        self.gpu_compute_time += time() - compute_start
        
        # transfer back
        transfer_start = time()
        offspring_flat = d_offspring.copy_to_host()
        self.gpu_transfer_time += time() - transfer_start
        
        return offspring_flat.reshape(num_offspring, self.gene_length).astype(np.float64)
    
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
    (10, 20, True),
    (10, 20, False),
    (50, 50, True),
    (50, 50, False),
    (100, 100, True),
    (100, 100, False),
    (500, 200, True),
    (500, 200, False),
    (1000, 1000, True),
    (1000, 1000, False),
    (1000, 2000, True),
    (1000, 2000, False),
]


@pytest.mark.parametrize("population_size, generations, gpu_enabled", TEST_PARAMS)
def test_perceptron_xor_ga(population_size, generations, gpu_enabled, logger: CUDA_GA_TestLogger):
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
        "gpu_enabled": gpu_enabled,
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
        gpu_enabled=gpu_enabled,
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
