from time import time
import math
import random
from numba import cuda
import pytest
import logging
import numpy as np

logger = logging.getLogger(__name__)

#! First attempt to optimize a third-party GA

# suppress verbose numba CUDA driver logs
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)

@cuda.jit
def calculate_fitness_kernel(paths, distance_matrix, fitnesses, pop_size, path_length, num_cities):
    # one thread computes the fitness of one path
    
    # - paths: array of paths (pop_size x path_length), values - city indexes
    # - distance_matrix: distance matrix (cities count x cities count)
    # - fitnesses: output array with path costs
    # - chrom_length = number of cities in the path
    
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    position = thread_id + block_id * block_width # type: ignore
    
    if position < pop_size:
        total_cost = 0.0 # target cost is minimum

        for i in range(path_length - 1):
            city_from = paths[position * path_length + i]
            city_to = paths[position * path_length + i + 1]
            total_cost += distance_matrix[city_from * num_cities + city_to]
        
        fitnesses[position] = total_cost


# Source of GA: https://itnext.io/the-genetic-algorithm-and-the-travelling-salesman-problem-tsp-31dfa57f3b62 


class Graph:
  def __init__(self, size, directed, start_city):
    self.nodes_len = size
    self.roots = {}
    self.nodes = {}
    self.start_city = start_city
    self.directed = directed
    
    self.distance_matrix = None
    self.city_to_idx = {}

  # connects node a and b
  def addEdge(self, a, b, weight=1):
    if a not in self.roots:
      self.roots[a] = []
    if b not in self.roots:
      self.roots[b] = []

    self.roots[a].append((b,weight))
    if not self.directed:
      self.roots[b].append((a,weight))

  # adds a node at X,Y
  # and also adds a direct edge to all other nodes
  # by calculating the euclidean distance
  def addNode(self, a, x, y):
    if a not in self.roots:
      self.nodes[a]= (x,y)
      for key, values in self.nodes.items():
        if key != a:
          distance = self.distance(a, key)
          self.addEdge(a, key, int(distance))

  # euclidean distance between two nodes
  def distance(self, a, b):
    X1 = self.nodes[a][0]
    X2 = self.nodes[b][0]
    Y1 = self.nodes[a][1]
    Y2 = self.nodes[b][1]
    distance = math.sqrt((X1-X2)**2 + (Y1-Y2)**2)
    return round(distance,2)

  def vertices(self):
    return list(self.roots.keys())

  # e.g PATH IS CAEDB
  def getPathCost(self, path, will_return=False):
    cost = 0
    for i in range(len(path)-1):
        cost += self.distance(path[i], path[i+1])
    if will_return:
       cost+= self.distance(path[0], path[-1])
    return cost

  def index_values(self):
      cities = sorted(self.vertices())
      n = len(cities)
      
      # project cities indexes to integers
      self.city_to_idx = {city: idx for idx, city in enumerate(cities)}
      
      # distance matrix [n x n]
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

    def __init__(self, generations=20, population_size=10, tournamentSize=4, mutationRate=0.1, fit_selection_rate=0.1, gpu_enabled=False):
        # the size of the population of routes 
        self.population_size = population_size
        # the number of generations to run the algorithm for
        self.generations = generations
        # the number of routes to select for crossover
        self.tournamentSize = tournamentSize
        self.mutationRate = mutationRate
        # The number of fittest routes to carry over to the next generation
        self.fit_selection_rate = fit_selection_rate
        self.gpu_enabled = gpu_enabled
        
        self.fitness_compute_time = 0
        self.total_fitness_calls = 0
        
    def minCostIndex(self, costs):
       index = 0
       for i in range(len(costs)):
            if costs[i] < costs[index]:
                index = i
       return index

    def find_fittest_path(self, graph):
        if self.gpu_enabled:
            graph.index_values() # convert to integers
            
        population = self.randomizeCities(graph, graph.vertices())
        # the number of fittest routes to carry over to the next generation
        number_of_fits_to_carryover = math.ceil(self.population_size * self.fit_selection_rate)

        if (number_of_fits_to_carryover > self.population_size):
            raise ValueError('fitness Rate must be in [0,1].')

        logger.info('Optimizing TSP Route for Graph:\n{0}'.format(graph))

        for generation in range(self.generations):
            logger.info('\nGeneration: {0}'.format(generation + 1))
            logger.info('Population: {0}'.format(', '.join(population[:10]) + '...'))

            newPopulation = []
            fitness = self.computeFitness(graph, population)
            fitIndex = self.minCostIndex(fitness)
            logger.info('Fittest Route: {0} fitness(minium cost): ({1})'.format(population[fitIndex], fitness[fitIndex]))

            # Add fit population to newPopulation
            if number_of_fits_to_carryover:
                sorted_population = [x for _,x in sorted(zip(fitness,population))]
                fitness = sorted(fitness)
                [newPopulation.append(sorted_population[i]) for i in range( number_of_fits_to_carryover)]
                # logger.info('sorted population: {0}, fitness:{1}\n newPpopulation={2}'.format(sorted_population, fitness, newPopulation))

            # create the remaining population
            # through crossover and mutation
            for gen in range(self.population_size-number_of_fits_to_carryover):
                parent1 = self.tournamentSelection(graph, population)
                parent2 = self.tournamentSelection(graph, population)
                # logger.info("parent1: {0}, parent2: {1}".format(parent1, parent2))
                offspring = self.crossover(parent1, parent2)
                newPopulation.append(offspring)
                # logger.info('Offspring: {0}\n'.format(offspring))
            # This is the mutation step
            for gen in range(self.population_size-number_of_fits_to_carryover):
                newPopulation[gen] = self.mutate(newPopulation[gen])

            population = newPopulation

            if self.converged(population):
                logger.info("converged", population)
                logger.info('\nConverged to a local minima.')
                break

        return (population[fitIndex], fitness[fitIndex])

    def randomizeCities(self, graph, graph_nodes):
       result= []
       # nodes without the start city
       nodes = [node for node in graph_nodes if node != graph.start_city]
       for i in range(self.population_size):
          random.shuffle(nodes)
          # add A as the first and last city
          cities = graph.start_city + ''.join(nodes) + graph.start_city
          result.append(cities)
       return result

    # Fitness is the cost of the path
    # lower the cost, higher the fitness
    def computeFitness(self, graph, population):
        start_time = time()
        self.total_fitness_calls += 1
        
        match (self.gpu_enabled):
            case True:
                fitness = self.computeFitnessGPU(graph, population)
            case False:
                fitness = self.computeFitnessCPU(graph, population)
                
        end_time = time()
        self.fitness_compute_time += end_time - start_time
    
        return fitness
    
    
    def computeFitnessCPU(self, graph, population):
        fitness = []
        for path in population:
            fitness.append(graph.getPathCost(path))
        return fitness

    def computeFitnessGPU(self, graph, population):
        pop_size = len(population)
        path_length = len(population[0])
        num_cities = len(graph.city_to_idx)
        
        # convert paths to indices array for easier access
        paths_indices = np.zeros((pop_size, path_length), dtype=np.int32)
        for i, path in enumerate(population):
            for j, city in enumerate(path):
                paths_indices[i, j] = graph.city_to_idx[city]
            
        
        # 1D array for CUDA
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

    # Chooses a parent with the best fitness from a random subset of the population
    # The subset is of size tournamentSize
    # e.g if tournamentSize = 4, then 4 random paths are chosen from the population
    # and the path with the lowest cost is chosen as the parent
    def tournamentSelection(self, graph, population):
        tournament_contestants = random.choices(population, k=self.tournamentSize)
        tournament_contestants_fitness = self.computeFitness(graph, tournament_contestants)
        return tournament_contestants[tournament_contestants_fitness.index(min(tournament_contestants_fitness))]

    # This method uses simple order crossover using two points to generate a new offspring
    # from two parents
    def crossover(self,parent1, parent2):
      offspring_length = len(parent1)-2 #excluding first and last city
      offspring = ['' for _ in range(offspring_length)]
      index_low, index_high = self.computeTwoPointIndexes(parent1)
      # Copy the genes from parent1 to the offspring
      offspring[index_low:index_high+1] = list(parent1)[index_low:index_high+1]
      # The remaining genes are copied from parent2
      empty_place_indexes = [i for i in range(offspring_length) if offspring[i] == '']
      for j in parent2[1:-1]:  # Exclude the start and end cities
          if '' not in offspring or not empty_place_indexes:
              break
          if j not in offspring:
              offspring[empty_place_indexes.pop(0)] = j

      offspring = ['A'] + offspring + ['A']
      return ''.join(offspring)


    def mutate(self, genome):
        if random.random() < self.mutationRate:
            index_low, index_high = self.computeTwoPointIndexes(genome)
            return self.swap(index_low, index_high, genome)
        else:
            return genome

    # selects indexes from parents such that the difference between the two indexes is less than half the length of the parent
    #  A B C D E F G H I J A
    #  0 1 2 3 4 5 6 7 8 9 10
    # index_low should be between 1 and 8
    # index_high should be between 2 and 9
    # because the start and end cities are fixed
    def computeTwoPointIndexes(self,parent):
      index_low = random.randint(1, len(parent)-3)
      index_high = random.randint(index_low+1, len(parent)-2)
      # make sure the difference between the two indexes is less than half the length of the parent
      if(index_high - index_low > math.ceil(len(parent)//2)):
          return self.computeTwoPointIndexes(parent)
      else:
          return index_low, index_high

    def swap(self, index_low, index_high, string):
        string = list(string)
        string[index_low], string[index_high] = string[index_high], string[index_low]
        return ''.join(string)

    # check if all the genomes in the population are the same
    # if so, then we have converged to a local minima
    def converged(self, population):
        return all(genome == population[0] for genome in population)



@pytest.mark.parametrize("gpu_enabled", [False, True])
def test_salesman_simple_ga(gpu_enabled):
    logger.info(f"Testing with GPU enabled: {gpu_enabled}")
    
    #! To test GPU optimization, we need to use a larger graph, 
    #! larger population, more generations 
    #! and possible more complicated calculations
    
    
    # This exact algorithm currently has lots of memory transfers and few computations per chromosome
    
    graph = Graph(10, False, 'A')
    graph.addNode('A', 100, 300)
    graph.addNode('B', 200,130)
    graph.addNode('C', 300,500)
    graph.addNode('D', 500, 390)
    graph.addNode('E', 700, 300)
    graph.addNode('F', 900, 600)
    graph.addNode('G', 800, 950)
    graph.addNode('H', 600, 560)
    graph.addNode('I', 350, 550)
    graph.addNode('J', 270, 350)
    
    population_size = graph.nodes_len * 10
    generations = 20
    tournamentSize = 5
    mutationRate = 0.1
    fit_selection_rate = 0.5
    
    logger.info(graph.showGraph())
    logger.info(f"Population size: {population_size}")
    logger.info(f"Generations: {generations}")
    logger.info(f"Tournament size: {tournamentSize}")
    logger.info(f"Mutation rate: {mutationRate}")
    logger.info(f"Fit selection rate: {fit_selection_rate}")

    ga_tsp = GeneticAlgorithmTSP(generations=generations, 
                                 population_size=population_size, 
                                 tournamentSize=tournamentSize, 
                                 mutationRate=mutationRate, 
                                 fit_selection_rate=fit_selection_rate, 
                                 gpu_enabled=gpu_enabled)

    fittest_path, path_cost = ga_tsp.find_fittest_path(graph)
    logger.info('\nPath: {0}, Cost: {1}'.format(fittest_path, path_cost))
    logger.info(f'Total fitness computation time: {ga_tsp.fitness_compute_time:.4f} seconds')
    logger.info(f'Total fitness calls: {ga_tsp.total_fitness_calls}')