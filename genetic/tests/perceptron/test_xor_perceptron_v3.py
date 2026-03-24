# pyright: reportCallIssue=false
#
# V3: + population on device

from time import time
from numba.core.types import logging
import numpy as np
from numpy._typing import NDArray
import pytest
from numba import cuda, float32, int32

from genetic.tests.logger import CUDA_GA_TestLogger
from genetic.tests.constants import RANDOM_SEED

numba_logger = logging.getLogger("numba.cuda.cudadrv.driver")
numba_logger.setLevel(logging.WARNING)


@cuda.jit
def fitness_kernel(population, inputs, outputs, fitnesses, pop_size, num_inputs, gene_length):
    shared_inputs = cuda.shared.array(shape=(8,), dtype=float32)  # type: ignore[call-arg]
    shared_outputs = cuda.shared.array(shape=(4,), dtype=int32)  # type: ignore[call-arg]

    tx = cuda.threadIdx.x
    if tx < num_inputs * gene_length:
        shared_inputs[tx] = inputs[tx]  # type: ignore
    if tx < num_inputs:
        shared_outputs[tx] = outputs[tx]  # type: ignore
    cuda.syncthreads()

    idx = cuda.grid(1)  # type: ignore
    if idx < pop_size:
        correct = 0
        base = idx * gene_length
        for i in range(num_inputs):
            dot_val = 0.0
            for j in range(gene_length):
                dot_val += population[base + j] * shared_inputs[i * gene_length + j]  # type: ignore
            pred = 1 if dot_val >= 0 else 0
            if pred == shared_outputs[i]:  # type: ignore[index]
                correct += 1
        fitnesses[idx] = correct


@cuda.jit
def selection_gather_kernel(population, sorted_indices, elite, elite_size, gene_length):
    idx = cuda.grid(1)  # type: ignore
    if idx < elite_size:
        src = sorted_indices[idx]
        for j in range(gene_length):
            elite[idx * gene_length + j] = population[src * gene_length + j]


@cuda.jit
def crossover_kernel(parents, offspring, crossover_points, num_offspring, gene_length):
    idx = cuda.grid(1)  # type: ignore
    if idx < num_offspring:
        p1_base = idx * 2 * gene_length
        p2_base = (idx * 2 + 1) * gene_length  # type: ignore
        out_base = idx * gene_length
        cp = crossover_points[idx]
        for i in range(cp):
            offspring[out_base + i] = parents[p1_base + i]
        for i in range(cp, gene_length):
            offspring[out_base + i] = parents[p2_base + i]


@cuda.jit
def mutation_kernel(population, mutation_values, mutation_flags, num_offspring, gene_length):
    idx = cuda.grid(1)  # type: ignore
    if idx < num_offspring and mutation_flags[idx] == 1:
        mut_idx = int(mutation_values[idx * 2 + 1])  # type: ignore
        population[idx * gene_length + mut_idx] += mutation_values[idx * 2]


@cuda.jit
def merge_population_kernel(elite, offspring, population, elite_size, num_offspring, gene_length):
    idx = cuda.grid(1)  # type: ignore
    total = elite_size + num_offspring
    if idx < total:
        if idx < elite_size:
            for j in range(gene_length):
                population[idx * gene_length + j] = elite[idx * gene_length + j]
        else:
            o_idx = idx - elite_size
            for j in range(gene_length):
                population[idx * gene_length + j] = offspring[o_idx * gene_length + j]


class PerceptronXOR:
    XOR_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    XOR_outputs = np.array([0, 1, 1, 0])

    def fitness(self, weights: NDArray[np.float64]) -> float:
        preds = [self.activation_function(weights, x) for x in self.XOR_inputs]
        return np.sum(preds == self.XOR_outputs)

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

        self.fitness_compute_time = 0.0
        self.gpu_transfer_time = 0.0
        self.gpu_compute_time = 0.0
        self.total_fitness_calls = 0

        if self.gpu_enabled:
            self._prepare_gpu_data()

    def _prepare_gpu_data(self):
        t0 = time()
        if self.seed is not None:
            np.random.seed(self.seed)
        elite_size = self.population_size // 2
        num_offspring = self.population_size // 2

        self.XOR_inputs_flat = self.perceptron.XOR_inputs.flatten().astype(np.float32)
        self.XOR_outputs_flat = self.perceptron.XOR_outputs.astype(np.int32)
        self.d_inputs = cuda.to_device(self.XOR_inputs_flat)
        self.d_outputs = cuda.to_device(self.XOR_outputs_flat)

        # persistent device buffers
        self.d_population = cuda.device_array(self.population_size * self.gene_length, dtype=np.float64)
        self.d_fitnesses = cuda.device_array(self.population_size, dtype=np.float64)
        self.d_elite = cuda.device_array(elite_size * self.gene_length, dtype=np.float64)
        self.d_offspring = cuda.device_array(num_offspring * self.gene_length, dtype=np.float64)
        self.d_sorted_indices = cuda.device_array(self.population_size, dtype=np.int32)  # type: ignore[arg-type]

        self.threads_per_block = 256
        self.blocks_pop = (self.population_size + self.threads_per_block - 1) // self.threads_per_block
        self.blocks_elite = (elite_size + self.threads_per_block - 1) // self.threads_per_block
        self.blocks_offspring = (num_offspring + self.threads_per_block - 1) // self.threads_per_block

        self.gpu_transfer_time += time() - t0
        self.logger.info("GPU prep complete (population resident across generations, seed set)")

    def _create_initial_population(self) -> NDArray[np.float64]:
        return np.random.uniform(-1, 1, (self.population_size, self.gene_length))

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        population = self._create_initial_population()
        if self.gpu_enabled:
            return self._run_gpu(population)
        return self._run_cpu(population)

    def _run_gpu(self, population: NDArray[np.float64]):
        elite_size = self.population_size // 2
        num_offspring = self.population_size // 2
        num_inputs = 4

        # initial transfer once
        t0 = time()
        self.d_population.copy_to_device(population.flatten())
        self.gpu_transfer_time += time() - t0

        for gen in range(self.generations):
            # fitness on GPU
            start = time()
            fitness_kernel[self.blocks_pop, self.threads_per_block](  # type: ignore
                self.d_population, self.d_inputs, self.d_outputs,
                self.d_fitnesses, self.population_size, num_inputs, self.gene_length
            )
            cuda.synchronize()
            self.gpu_compute_time += time() - start
            self.total_fitness_calls += 1

            # bring fitnesses for sorting
            t0 = time()
            fitnesses = self.d_fitnesses.copy_to_host()
            self.gpu_transfer_time += time() - t0

            sorted_indices = np.argsort(fitnesses)[::-1].astype(np.int32)
            best_fitness = fitnesses[sorted_indices[0]]

            # send sorted indices for device gather
            t0 = time()
            self.d_sorted_indices.copy_to_device(sorted_indices)
            self.gpu_transfer_time += time() - t0

            # gather elite on GPU
            start = time()
            selection_gather_kernel[self.blocks_elite, self.threads_per_block](  # type: ignore
                self.d_population, self.d_sorted_indices, self.d_elite,
                elite_size, self.gene_length
            )

            # host prepares parents/mutation data
            elite_host = self.d_elite.copy_to_host()
            parent_pairs = []
            for _ in range(num_offspring):
                p = np.random.choice(elite_size, 2, replace=False)
                parent_pairs.extend([elite_host[p[0]], elite_host[p[1]]])
            parent_pairs_array = np.array(parent_pairs, dtype=np.float64).flatten()

            crossover_points = np.random.randint(1, self.gene_length, num_offspring, dtype=np.int32)
            mutation_flags = (np.random.rand(num_offspring) < self.mutation_rate).astype(np.int32)
            mutation_indices = np.random.randint(0, self.gene_length, num_offspring, dtype=np.int32)
            mutation_vals = np.random.normal(0, 1, num_offspring).astype(np.float64)
            mutation_data = np.zeros(num_offspring * 2, dtype=np.float64)
            mutation_data[::2] = mutation_vals
            mutation_data[1::2] = mutation_indices

            # transfer parents/mutation
            t0 = time()
            d_parents = cuda.to_device(parent_pairs_array)
            d_crossover_points = cuda.to_device(crossover_points)
            d_mutation_data = cuda.to_device(mutation_data)
            d_mutation_flags = cuda.to_device(mutation_flags)
            self.gpu_transfer_time += time() - t0

            # crossover + mutation on GPU
            crossover_kernel[self.blocks_offspring, self.threads_per_block](  # type: ignore
                d_parents, self.d_offspring, d_crossover_points,
                num_offspring, self.gene_length
            )
            mutation_kernel[self.blocks_offspring, self.threads_per_block](  # type: ignore
                self.d_offspring, d_mutation_data, d_mutation_flags,
                num_offspring, self.gene_length
            )

            # merge back
            merge_population_kernel[self.blocks_pop, self.threads_per_block](  # type: ignore
                self.d_elite, self.d_offspring, self.d_population,
                elite_size, num_offspring, self.gene_length
            )
            cuda.synchronize()
            self.gpu_compute_time += time() - start

            if gen % 100 == 0 or gen == self.generations - 1:
                self.logger.info(f"Generation {gen}, best fitness: {best_fitness}, gpu_enabled: True")
            if best_fitness == 4:
                self.logger.info(f"Perfect solution found at generation {gen}!")
                break

        t0 = time()
        pop_host = self.d_population.copy_to_host()
        self.gpu_transfer_time += time() - t0
        best_weights = pop_host[:self.gene_length].copy()

        self.logger.info(f"Best weights: {best_weights}")
        self.logger.info(f"GPU transfer time: {self.gpu_transfer_time:.4f}s")
        self.logger.info(f"GPU compute time: {self.gpu_compute_time:.4f}s")
        self.logger.info(f"Total fitness calls: {self.total_fitness_calls}")
        return best_weights

    def _run_cpu(self, population: NDArray[np.float64]):
        for gen in range(self.generations):
            start = time()
            fitnesses = np.array([self.perceptron.fitness(w) for w in population])
            self.fitness_compute_time += time() - start
            self.total_fitness_calls += 1

            sorted_indices = np.argsort(fitnesses)[::-1]
            population = population[sorted_indices]
            elite = population[: self.population_size // 2].copy()

            offspring = self._crossover_mutation_cpu(elite)
            population = np.vstack((elite, offspring))
            best_fitness = fitnesses[sorted_indices[0]]

            if gen % 100 == 0 or gen == self.generations - 1:
                self.logger.info(f"Generation {gen}, best fitness: {best_fitness}, gpu_enabled: False")
            if best_fitness == 4:
                self.logger.info(f"Perfect solution found at generation {gen}!")
                break

        best_weights = population[0]
        self.logger.info(f"Best weights: {best_weights}")
        self.logger.info(f"Total fitness compute time: {self.fitness_compute_time:.4f}s")
        return best_weights

    def _crossover_mutation_cpu(self, elite: NDArray[np.float64]) -> NDArray[np.float64]:
        offspring = []
        while len(offspring) < self.population_size // 2:
            p = np.random.choice(len(elite), 2, replace=False)
            child = self._crossover(elite[p[0]], elite[p[1]])
            child = self._mutate(child)
            offspring.append(child)
        return np.array(offspring)

    def _crossover(self, parent1: NDArray[np.float64], parent2: NDArray[np.float64]) -> NDArray[np.float64]:
        cp = np.random.randint(1, self.gene_length)
        return np.concatenate([parent1[:cp], parent2[cp:]])

    def _mutate(self, child: NDArray[np.float64]) -> NDArray[np.float64]:
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(self.gene_length)
            child[idx] += np.random.normal()
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
    test_logger = logger.logger
    test_logger.info("\n" + "=" * 80)
    test_logger.info("NEW TEST RUN")
    test_logger.info(f"Population size: {population_size}")
    test_logger.info(f"Generations: {generations}")

    mutation_rate = 0.1
    gene_length = 2

    params = {
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "gene_length": gene_length,
        "gpu_enabled": gpu_enabled,
    }

    logger.start_test(f"test_xor_ga_pop_{population_size}_gens_{generations}")
    logger.log_params(params)

    ga = GA_PerceptronXOR(
        logger=logger,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        gene_length=gene_length,
        gpu_enabled=gpu_enabled,
    )

    best_weights = ga.run()

    test_logger.info("Testing perceptron with best weights.")
    perceptron = ga.perceptron
    predictions = [perceptron.activation_function(best_weights, x) for x in perceptron.XOR_inputs]  # type: ignore[arg-type]
    test_logger.info(f"Predictions: {predictions}")
    test_logger.info(f"Expected: {perceptron.XOR_outputs.tolist()}")
    final_test_fitness = sum(int(p == t) for p, t in zip(predictions, perceptron.XOR_outputs))

    passed = final_test_fitness >= 2
    assert passed, f"Perceptron cannot fully solve XOR; expected fitness >=2, got {final_test_fitness}"

    test_logger.info("XOR test passed with best weights.")
    test_logger.info("GA run completed. Test passed.")
    logger.end_test("passed" if passed else "failed", extra_info={"best_weights": best_weights})

