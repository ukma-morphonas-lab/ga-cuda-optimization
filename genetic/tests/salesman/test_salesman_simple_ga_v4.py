# V5: + pinned arrays + GPU RNG for mutation + GPU mutation kernel

from time import time
import math
import random
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import pytest
import numpy as np
import logging

from genetic.tests.logger import CUDA_GA_TestLogger
from genetic.tests.constants import RANDOM_SEED

numba_logger = logging.getLogger("numba.cuda.cudadrv.driver")
numba_logger.setLevel(logging.WARNING)

# V4:+ pre-allocated arrays  + pinned host buffers + GPU RNG for mutation

@cuda.jit
def calculate_fitness_kernel(paths, distance_matrix, fitnesses, pop_size, path_length, num_cities):
    idx = cuda.grid(1)  # type: ignore
    if idx < pop_size:
        total_cost = 0.0
        base = idx * path_length
        for i in range(path_length - 1):
            city_from = paths[base + i]
            city_to = paths[base + i + 1]
            total_cost += distance_matrix[city_from * num_cities + city_to]
        fitnesses[idx] = total_cost


@cuda.jit
def mutation_kernel(paths, rng_states, pop_size, path_length, mutation_rate):
    """GPU swap mutation for TSP paths.
    
    Swaps two random cities (excluding first/last which are start city).
    """
    idx = cuda.grid(1)  # type: ignore
    if idx < pop_size:
        mutation_flag = xoroshiro128p_uniform_float32(rng_states, idx) < mutation_rate  # type: ignore
        if not mutation_flag:
            return
        
        base = idx * path_length
        
        # 2 random indices in range [1, path_length-2] (excluding start/end)
        inner_len = path_length - 2  # excludes first and last
        i1 = int(xoroshiro128p_uniform_float32(rng_states, idx) * inner_len) + 1  # type: ignore
        i2 = int(xoroshiro128p_uniform_float32(rng_states, idx) * inner_len) + 1  # type: ignore
        
        if i1 == i2:
            i2 = (i2 % inner_len) + 1
            if i2 == i1:
                i2 = ((i2 + 1) % inner_len) + 1
                
        # swap
        tmp = paths[base + i1]
        paths[base + i1] = paths[base + i2]
        paths[base + i2] = tmp


class Graph:
    def __init__(self, size, directed, custom_logger: CUDA_GA_TestLogger, start_city=0):
        self.logger = custom_logger.logger
        self.nodes_len = size
        self.roots = {}
        self.nodes = {}
        self.start_city = start_city
        self.directed = directed
        self.distance_matrix = None
        self.d_distance_matrix = None

    def addEdge(self, a, b, weight=1):
        if a not in self.roots:
            self.roots[a] = []
        if b not in self.roots:
            self.roots[b] = []

        self.roots[a].append((b, weight))
        if not self.directed:
            self.roots[b].append((a, weight))

    def addNode(self, city_id, x, y):
        if city_id not in self.roots:
            self.nodes[city_id] = (x, y)
            for existing_city in self.nodes.keys():
                if existing_city != city_id:
                    distance = self.distance(city_id, existing_city)
                    self.addEdge(city_id, existing_city, int(distance))

    def distance(self, a, b):
        x1, y1 = self.nodes[a]
        x2, y2 = self.nodes[b]
        return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 2)

    def vertices(self):
        return sorted(list(self.roots.keys()))

    def getPathCost(self, path, will_return=False):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.distance(path[i], path[i + 1])
        if will_return:
            cost += self.distance(path[0], path[-1])
        return cost

    def index_values(self):
        cities = self.vertices()
        n = len(cities)

        self.distance_matrix = np.zeros((n, n), dtype=np.float32)
        for i, city_i in enumerate(cities):
            for j, city_j in enumerate(cities):
                if i != j:
                    self.distance_matrix[i, j] = self.distance(city_i, city_j)

        self.d_distance_matrix = cuda.to_device(self.distance_matrix.flatten())
        self.logger.info(f"GPU prep complete: {n} cities, distance matrix shape: {self.distance_matrix.shape}")

    def showGraph(self):
        self.logger.info("Graph connections:")
        for city, connections in self.roots.items():
            connections_str = ", ".join([f"{dest}({dist})" for dest, dist in connections])
            self.logger.info(f"City {city}: {connections_str}")
        return f"Graph with {len(self.roots)} cities"


class GeneticAlgorithmTSP:
    def __init__(
        self,
        custom_logger: CUDA_GA_TestLogger,
        generations=20,
        population_size=10,
        tournamentSize=4,
        mutationRate=0.1,
        fit_selection_rate=0.1,
        gpu_enabled=False,
        seed=RANDOM_SEED,
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournamentSize = tournamentSize
        self.mutationRate = mutationRate
        self.fit_selection_rate = fit_selection_rate
        self.gpu_enabled = gpu_enabled
        self.logger = custom_logger.logger
        self.seed = seed

        self.fitness_compute_time = 0.0
        self.total_fitness_calls = 0
        self.gpu_transfer_time = 0.0
        self.gpu_compute_time = 0.0

        # device buffers
        self.d_paths = None
        self.d_fitnesses = None
        self.d_offspring = None
        # pinned host buffers
        self.h_paths = None
        self.h_fitnesses = None
        self.h_offspring = None
        # GPU RNG states
        self.rng_states = None

        self.path_length = None
        self.num_cities = None
        self.num_offspring = None
        self.threads_per_block = 256
        self.blocks_per_grid = None
        self.blocks_offspring = None

    def minCostIndex(self, costs):
        return min(range(len(costs)), key=lambda i: costs[i])

    def find_fittest_path(self, graph):
        if self.gpu_enabled:
            graph.index_values()

        population = self.randomizeCities(graph, graph.vertices())
        number_of_fits_to_carryover = math.ceil(self.population_size * self.fit_selection_rate)

        if number_of_fits_to_carryover > self.population_size:
            raise ValueError("fitness Rate must be in [0,1].")

        if self.gpu_enabled:
            self._prepare_gpu_buffers(graph, population)

        self.logger.info(f"Optimizing TSP Route for Graph: {graph}")

        for generation in range(self.generations):
            fitness = self.computeFitness(graph, population)
            fitness_array = np.array(fitness) 
            fitIndex = int(np.argmin(fitness_array))

            newPopulation = []

            if number_of_fits_to_carryover:
                sorted_indices = np.argsort(fitness_array)
                newPopulation.extend([population[i] for i in sorted_indices[:number_of_fits_to_carryover]])

            offspring_candidates = []
            for _ in range(self.population_size - number_of_fits_to_carryover):
                parent1 = self.tournamentSelection(population, fitness, fitness_array)
                parent2 = self.tournamentSelection(population, fitness, fitness_array)
                offspring_candidates.append(self.crossover(parent1, parent2))

            # Mutation: GPU or CPU
            if self.gpu_enabled:
                offspring_candidates = self.mutateGPU(offspring_candidates)
            else:
                for i in range(len(offspring_candidates)):
                    offspring_candidates[i] = self.mutateCPU(offspring_candidates[i])

            newPopulation.extend(offspring_candidates)
            population = newPopulation

            if generation % 5 == 0:
                self.logger.info(
                    f"Gen {generation + 1}: best cost {fitness[fitIndex]:.2f}, gpu_enabled={self.gpu_enabled}"
                )

            if self.converged(population):
                self.logger.info("Converged to a local minima.")
                break

        final_fitness = self.computeFitness(graph, population)
        fitIndex = self.minCostIndex(final_fitness)
        return (population[fitIndex], final_fitness[fitIndex])

    def randomizeCities(self, graph, graph_nodes):
        result = []
        nodes = [node for node in graph_nodes if node != graph.start_city]

        for _ in range(self.population_size):
            shuffled = nodes.copy()
            random.shuffle(shuffled)
            route = [graph.start_city] + shuffled + [graph.start_city]
            result.append(route)
        return result

    def computeFitness(self, graph, population):
        self.total_fitness_calls += 1

        if self.gpu_enabled:
            fitness = self.computeFitnessGPU(graph, population)
        else:
            start_time = time()
            fitness = self.computeFitnessCPU(graph, population)
            self.fitness_compute_time += time() - start_time

        return fitness

    def computeFitnessCPU(self, graph, population):
        return [graph.getPathCost(path) for path in population]

    def computeFitnessGPU(self, graph, population):
        if self.d_paths is None or self.d_fitnesses is None:
            raise RuntimeError("GPU buffers not prepared. Call _prepare_gpu_buffers first.")
        if self.h_paths is None or self.h_fitnesses is None:
            raise RuntimeError("Pinned host buffers not prepared.")

        # copy population into pinned buffer
        np.copyto(self.h_paths, np.asarray(population, dtype=np.int32).ravel())

        transfer_start = time()
        self.d_paths.copy_to_device(self.h_paths)
        self.gpu_transfer_time += time() - transfer_start

        compute_start = time()
        calculate_fitness_kernel[self.blocks_per_grid, self.threads_per_block](  # type: ignore
            self.d_paths,
            graph.d_distance_matrix,
            self.d_fitnesses,
            self.population_size,
            self.path_length,
            self.num_cities,
        )
        cuda.synchronize()
        self.gpu_compute_time += time() - compute_start
        self.fitness_compute_time += time() - compute_start

        transfer_start = time()
        self.d_fitnesses.copy_to_host(self.h_fitnesses)
        self.gpu_transfer_time += time() - transfer_start

        return [float(f) for f in self.h_fitnesses]

    def mutateGPU(self, offspring_list):
        num_offspring = len(offspring_list)
        if num_offspring == 0:
            return offspring_list

        if self.h_offspring is None or self.d_offspring is None or self.path_length is None:
            raise RuntimeError("GPU buffers not prepared for mutation.")

        # copy into pre-allocated pinned buffer
        np.copyto(self.h_offspring, np.asarray(offspring_list, dtype=np.int32).ravel())

        transfer_start = time()
        self.d_offspring.copy_to_device(self.h_offspring)
        self.gpu_transfer_time += time() - transfer_start

        compute_start = time()
        mutation_kernel[self.blocks_offspring, self.threads_per_block](  # type: ignore
            self.d_offspring,
            self.rng_states,
            num_offspring,
            self.path_length,
            self.mutationRate,
        )
        cuda.synchronize()
        self.gpu_compute_time += time() - compute_start

        transfer_start = time()
        self.d_offspring.copy_to_host(self.h_offspring)
        self.gpu_transfer_time += time() - transfer_start

        path_len = self.path_length
        return [list(self.h_offspring[i * path_len : (i + 1) * path_len]) for i in range(num_offspring)]

    def tournamentSelection(self, population, fitness_values, fitness_array=None):
        pop_len = len(population)
        tournament_indices = random.sample(range(pop_len), k=min(self.tournamentSize, pop_len))

        if fitness_array is not None:
            winner_local = np.argmin(fitness_array[tournament_indices])
        else:
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            winner_local = tournament_fitness.index(min(tournament_fitness))
        return population[tournament_indices[winner_local]]

    def crossover(self, parent1, parent2):
        offspring_length = len(parent1) - 2
        offspring = [None] * offspring_length
        index_low, index_high = self.computeTwoPointIndexes(parent1)

        for i in range(index_low, index_high + 1):
            offspring[i - 1] = parent1[i]

        # set for O(1) lookup instead of O(n) list membership
        offspring_set = set(offspring)
        parent2_cities = [city for city in parent2[1:-1] if city not in offspring_set]
        offspring_idx = 0

        for city in parent2_cities:
            while offspring_idx < offspring_length and offspring[offspring_idx] is not None:
                offspring_idx += 1
            if offspring_idx < offspring_length:
                offspring[offspring_idx] = city

        if None in offspring:
            used_cities = set(c for c in offspring if c is not None)
            missing_cities = list(set(parent1[1:-1]) - used_cities)

            for i in range(offspring_length):
                if offspring[i] is None and missing_cities:
                    offspring[i] = missing_cities.pop(0)

        return [parent1[0]] + offspring + [parent1[-1]]

    def mutateCPU(self, genome):
        if random.random() < self.mutationRate:
            index_low, index_high = self.computeTwoPointIndexes(genome)
            genome = genome.copy()
            genome[index_low], genome[index_high] = genome[index_high], genome[index_low]
        return genome

    def computeTwoPointIndexes(self, parent):
        index_low = random.randint(1, len(parent) - 3)
        index_high = random.randint(index_low + 1, len(parent) - 2)

        if index_high - index_low > math.ceil(len(parent) // 2):
            return self.computeTwoPointIndexes(parent)
        return index_low, index_high

    def converged(self, population):
        return all(genome == population[0] for genome in population)

    def _prepare_gpu_buffers(self, graph, population):
        self.path_length = len(population[0])
        self.num_cities = len(graph.vertices())
        self.num_offspring = self.population_size - math.ceil(self.population_size * self.fit_selection_rate)

        total_path_elems = self.population_size * self.path_length
        offspring_elems = self.num_offspring * self.path_length
        t0 = time()

        # Pinned host buffers for faster H<->D transfers
        self.h_paths = cuda.pinned_array(total_path_elems, dtype=np.int32)  # type: ignore[arg-type]
        self.h_fitnesses = cuda.pinned_array(self.population_size, dtype=np.float64)
        self.h_offspring = cuda.pinned_array(offspring_elems, dtype=np.int32)  # type: ignore[arg-type]

        # Device buffers
        self.d_paths = cuda.device_array(total_path_elems, dtype=np.int32)  # type: ignore[arg-type]
        self.d_fitnesses = cuda.device_array(self.population_size, dtype=np.float64)
        self.d_offspring = cuda.device_array(offspring_elems, dtype=np.int32)  # type: ignore[arg-type]

        # GPU RNG states for mutation
        self.rng_states = create_xoroshiro128p_states(self.population_size, seed=self.seed)

        self.blocks_per_grid = (self.population_size + self.threads_per_block - 1) // self.threads_per_block
        self.blocks_offspring = (self.num_offspring + self.threads_per_block - 1) // self.threads_per_block
        self.gpu_transfer_time += time() - t0
        self.logger.info(
            f"GPU buffers prepared (pinned + RNG): pop={self.population_size}, path_len={self.path_length}, num_cities={self.num_cities}, offspring={self.num_offspring}"
        )


TEST_PARAMS = [
    (False, 10, 100),
    (True, 10, 100),
    (False, 100, 5000),
    (True, 100, 5000),
    (False, 200, 5000),
    (True, 200, 5000),
    (True, 500, 10000),
    (True, 500, 20000),
]

@pytest.mark.parametrize("gpu_enabled, graph_size, population_size", TEST_PARAMS)
def test_salesman_simple_ga(gpu_enabled, graph_size, population_size, logger: CUDA_GA_TestLogger):
    logger.logger.info("\n" + "=" * 80)
    logger.logger.info("NEW TEST RUN (V4 - pinned + GPU RNG)")
    logger.logger.info(f"Testing with GPU enabled: {gpu_enabled}")

    graph = create_random_graph(graph_size, logger)

    generations = 20
    tournamentSize = 5
    mutationRate = 0.1
    fit_selection_rate = 0.5

    logger.logger.info(f"Population size: {population_size}")
    logger.logger.info(f"Generations: {generations}")

    params = {
        "generations": generations,
        "population_size": population_size,
        "tournamentSize": tournamentSize,
        "mutationRate": mutationRate,
        "fit_selection_rate": fit_selection_rate,
        "gpu_enabled": gpu_enabled,
        "graph_size": graph_size,
    }

    logger.start_test(f"test_salesman_simple_ga_v4_gpu_{gpu_enabled}_size_{graph_size}_pop_{population_size}")
    logger.log_params(params)

    ga_tsp = GeneticAlgorithmTSP(
        custom_logger=logger,
        generations=generations,
        population_size=population_size,
        tournamentSize=tournamentSize,
        mutationRate=mutationRate,
        fit_selection_rate=fit_selection_rate,
        gpu_enabled=gpu_enabled,
    )

    fittest_path, path_cost = ga_tsp.find_fittest_path(graph)
    log_test_results(graph_size, fittest_path, path_cost, ga_tsp, gpu_enabled, logger)


def create_random_graph(num_cities, custom_logger: CUDA_GA_TestLogger, width=1000, height=1000, start_city=0):
    graph = Graph(num_cities, False, custom_logger=custom_logger, start_city=start_city)
    for city_id in range(num_cities):
        x = random.randint(0, width)
        y = random.randint(0, height)
        graph.addNode(city_id, x, y)
    return graph


def log_test_results(
    graph_size: int,
    fittest_path: list,
    path_cost: float,
    ga_tsp: GeneticAlgorithmTSP,
    gpu_enabled: bool,
    logger: CUDA_GA_TestLogger,
) -> None:
    output = f"""Graph size: {graph_size} cities
                Best path: {fittest_path}
                Path cost: {path_cost}
                Total fitness computation time: {ga_tsp.fitness_compute_time:.4f} seconds
                Total fitness calls: {ga_tsp.total_fitness_calls}
                Average time per fitness call: {ga_tsp.fitness_compute_time/ga_tsp.total_fitness_calls:.6f} seconds"""

    if gpu_enabled:
        output += f"""
                GPU transfer time: {ga_tsp.gpu_transfer_time:.4f} seconds
                GPU compute time: {ga_tsp.gpu_compute_time:.4f} seconds
                Transfer overhead: {(ga_tsp.gpu_transfer_time/ga_tsp.fitness_compute_time)*100:.1f}%"""

    logger.log_output(output)

    logger.logger.info("\nPath: {0}, Cost: {1}".format(fittest_path, path_cost))
    logger.logger.info(f"Total fitness computation time: {ga_tsp.fitness_compute_time:.4f} seconds")
    logger.logger.info(f"Total fitness calls: {ga_tsp.total_fitness_calls}")

    logger.end_test("passed")

