from time import time
import math
import random
from numba import cuda
import pytest
import logging
import numpy as np

logger = logging.getLogger(__name__)

# suppress verbose numba CUDA driver logs
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)

@cuda.jit
def calculate_fitness_kernel(paths, distance_matrix, fitnesses, pop_size, path_length, num_cities):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    position = thread_id + block_id * block_width # type: ignore
    
    if position < pop_size:
        total_cost = 0.0
        for i in range(path_length - 1):
            city_from = paths[position * path_length + i]
            city_to = paths[position * path_length + i + 1]
            total_cost += distance_matrix[city_from * num_cities + city_to]
        
        fitnesses[position] = total_cost


class Graph:
    def __init__(self, size, directed, start_city=0):
        self.nodes_len = size
        self.roots = {}
        self.nodes = {}
        self.start_city = start_city
        self.directed = directed
        self.distance_matrix = None

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
        X1, Y1 = self.nodes[a]
        X2, Y2 = self.nodes[b]
        distance = math.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
        return round(distance, 2)

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
        
        logger.info(f"GPU preparation complete: {n} cities, distance matrix shape: {self.distance_matrix.shape}")

    def showGraph(self):
        logger.info("Graph connections:")
        for city, connections in self.roots.items():
            connections_str = ", ".join([f"{dest}({dist})" for dest, dist in connections])
            logger.info(f"City {city}: {connections_str}")
        return f"Graph with {len(self.roots)} cities"


class GeneticAlgorithmTSP:
    def __init__(self, generations=20, population_size=10, tournamentSize=4, 
                 mutationRate=0.1, fit_selection_rate=0.1, gpu_enabled=False):
        self.population_size = population_size
        self.generations = generations
        self.tournamentSize = tournamentSize
        self.mutationRate = mutationRate
        self.fit_selection_rate = fit_selection_rate
        self.gpu_enabled = gpu_enabled
        
        self.fitness_compute_time = 0
        self.total_fitness_calls = 0
        
    def minCostIndex(self, costs):
        return min(range(len(costs)), key=lambda i: costs[i])

    def find_fittest_path(self, graph):
        if self.gpu_enabled:
            graph.index_values()
            
        population = self.randomizeCities(graph, graph.vertices())
        number_of_fits_to_carryover = math.ceil(self.population_size * self.fit_selection_rate)

        if number_of_fits_to_carryover > self.population_size:
            raise ValueError('fitness Rate must be in [0,1].')

        logger.info('Optimizing TSP Route for Graph:\n{0}'.format(graph))

        for generation in range(self.generations):
            logger.info('\nGeneration: {0}'.format(generation + 1))
            logger.info('Population sample: {0}'.format(str(population[:3]) + '...'))

            newPopulation = []
            fitness = self.computeFitness(graph, population)
            fitIndex = self.minCostIndex(fitness)
            logger.info('Fittest Route: {0} fitness(minimum cost): ({1})'.format(
                population[fitIndex], fitness[fitIndex]))

            # Add fit population to newPopulation
            if number_of_fits_to_carryover:
                sorted_population = [x for _, x in sorted(zip(fitness, population))]
                fitness = sorted(fitness)
                newPopulation.extend(sorted_population[:number_of_fits_to_carryover])

            # Create remaining population through crossover
            for _ in range(self.population_size - number_of_fits_to_carryover):
                parent1 = self.tournamentSelection(graph, population)
                parent2 = self.tournamentSelection(graph, population)
                offspring = self.crossover(parent1, parent2)
                newPopulation.append(offspring)

            # Mutation step
            for gen in range(self.population_size - number_of_fits_to_carryover):
                newPopulation[gen] = self.mutate(newPopulation[gen])

            population = newPopulation

            if self.converged(population):
                logger.info("Converged to a local minima.")
                break

        return (population[fitIndex], fitness[fitIndex])

    def randomizeCities(self, graph, graph_nodes):
        result = []
        nodes = [node for node in graph_nodes if node != graph.start_city]
        
        for _ in range(self.population_size):
            shuffled = nodes.copy()
            random.shuffle(shuffled)
            # Add start city at beginning and end
            route = [graph.start_city] + shuffled + [graph.start_city]
            result.append(route)
        return result

    def computeFitness(self, graph, population):
        start_time = time()
        self.total_fitness_calls += 1
        
        if self.gpu_enabled:
            fitness = self.computeFitnessGPU(graph, population)
        else:
            fitness = self.computeFitnessCPU(graph, population)
                
        end_time = time()
        self.fitness_compute_time += end_time - start_time
        return fitness
    
    def computeFitnessCPU(self, graph, population):
        return [graph.getPathCost(path) for path in population]

    def computeFitnessGPU(self, graph, population):
        pop_size = len(population)
        path_length = len(population[0])
        num_cities = len(graph.vertices())
        
        # Convert paths to numpy array explicitly
        paths_indices = np.zeros((pop_size, path_length), dtype=np.int32)
        for i, path in enumerate(population):
            for j, city in enumerate(path):
                paths_indices[i, j] = city
        
        paths_flat = paths_indices.flatten()
        distance_matrix_flat = graph.distance_matrix.flatten()
        fitnesses = np.zeros(pop_size, dtype=np.float32)
        
        # CUDA threads setup
        threads_per_block = 256
        blocks_per_grid = (pop_size + threads_per_block - 1) // threads_per_block
        
        calculate_fitness_kernel[blocks_per_grid, threads_per_block]( # type: ignore
            paths_flat, distance_matrix_flat, fitnesses, 
            pop_size, path_length, num_cities
        )

        return [float(f) for f in fitnesses]

    def tournamentSelection(self, graph, population):
        tournament_contestants = random.choices(population, k=self.tournamentSize)
        tournament_fitness = self.computeFitness(graph, tournament_contestants)
        return tournament_contestants[tournament_fitness.index(min(tournament_fitness))]

    def crossover(self, parent1, parent2):
        offspring_length = len(parent1) - 2  # Excluding first and last city
        offspring = [None] * offspring_length
        index_low, index_high = self.computeTwoPointIndexes(parent1)
        
        # Copy segment from parent1 (adjust indices to exclude start city)
        for i in range(index_low, index_high + 1):
            offspring[i - 1] = parent1[i]
        
        # Fill remaining positions from parent2 in order
        parent2_cities = [city for city in parent2[1:-1] if city not in offspring]
        offspring_idx = 0
        
        for city in parent2_cities:
            while offspring_idx < offspring_length and offspring[offspring_idx] is not None:
                offspring_idx += 1
            if offspring_idx < offspring_length:
                offspring[offspring_idx] = city
        
        # Safety check: filling any remaining None values with missing cities
        if None in offspring:
            all_cities = set(parent1[1:-1])
            used_cities = set(c for c in offspring if c is not None)
            missing_cities = list(all_cities - used_cities)
            
            for i in range(offspring_length):
                if offspring[i] is None and missing_cities:
                    offspring[i] = missing_cities.pop(0)

        return [parent1[0]] + offspring + [parent1[-1]]

    def mutate(self, genome):
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


def create_random_graph(num_cities, width=1000, height=1000, start_city=0):
    graph = Graph(num_cities, False, start_city)
    for city_id in range(num_cities):
        x = random.randint(0, width)
        y = random.randint(0, height)
        graph.addNode(city_id, x, y)
    return graph


@pytest.mark.parametrize("gpu_enabled, graph_size", [(False, 10), (True, 10), (False, 100), (True, 100), (False, 500), (True, 500)])
def test_salesman_simple_ga(gpu_enabled, graph_size, logger):
    logger.logger.info(f"Testing with GPU enabled: {gpu_enabled}")

    graph = create_random_graph(graph_size)

    population_size = graph.nodes_len * 10
    generations = 20
    tournamentSize = 5
    mutationRate = 0.1
    fit_selection_rate = 0.5

    logger.logger.info(f"Population size: {population_size}")
    logger.logger.info(f"Generations: {generations}")

    # Log test parameters
    params = {
        'generations': generations,
        'population_size': population_size,
        'tournamentSize': tournamentSize,
        'mutationRate': mutationRate,
        'fit_selection_rate': fit_selection_rate,
        'gpu_enabled': gpu_enabled,
        'graph_size': graph_size
    }

    logger.start_test(f"test_salesman_simple_ga_v2_gpu_{gpu_enabled}_size_{graph_size}")
    logger.log_params(params)

    ga_tsp = GeneticAlgorithmTSP(
        generations=generations,
        population_size=population_size,
        tournamentSize=tournamentSize,
        mutationRate=mutationRate,
        fit_selection_rate=fit_selection_rate,
        gpu_enabled=gpu_enabled
    )

    fittest_path, path_cost = ga_tsp.find_fittest_path(graph)

    # Log results
    output = f"""Graph size: {graph_size} cities
Best path: {fittest_path}
Path cost: {path_cost}
Total fitness computation time: {ga_tsp.fitness_compute_time:.4f} seconds
Total fitness calls: {ga_tsp.total_fitness_calls}
Average time per fitness call: {ga_tsp.fitness_compute_time/ga_tsp.total_fitness_calls:.6f} seconds"""

    logger.log_output(output)

    logger.logger.info('\nPath: {0}, Cost: {1}'.format(fittest_path, path_cost))
    logger.logger.info(f'Total fitness computation time: {ga_tsp.fitness_compute_time:.4f} seconds')
    logger.logger.info(f'Total fitness calls: {ga_tsp.total_fitness_calls}')

    # End test successfully
    logger.end_test("passed")