from time import time
from typing import List, Tuple, Any
import logging
import numpy as np
import os

from genetic.tests.statistical_analysis_base import (
    run_full_analysis_generic,
    ExperimentConfig,
    RunResult,
)
from genetic.tests.fitness_metrics import MinimizationMetric
from genetic.tests.null_logger import create_null_logger
from genetic.tests.salesman.test_salesman_simple_ga_v4 import GeneticAlgorithmTSP, create_random_graph

logger = logging.getLogger(__name__)


def run_single_tsp(
    config: ExperimentConfig,
    gpu_enabled: bool,
    seed: int,
) -> RunResult:
    silent_logger = create_null_logger()
    start_wall = time()

    graph_size = config.extra_params.get('graph_size', 50)
    
    # fixed graph seed to reduce inter-graph variance
    # only vary GA initialization with seed parameter
    graph_seed = config.extra_params.get('graph_seed', 42)
    np.random.seed(graph_seed)
    graph = create_random_graph(graph_size, silent_logger, start_city=0)

    ga_tsp = GeneticAlgorithmTSP(
        custom_logger=silent_logger,
        generations=config.max_generations,
        population_size=config.population_size,
        tournamentSize=config.extra_params.get('tournament_size', 5),
        mutationRate=config.extra_params.get('mutation_rate', 0.1),
        fit_selection_rate=config.extra_params.get('fit_selection_rate', 0.5),
        gpu_enabled=gpu_enabled,
        seed=seed,
    )

    best_path, path_cost = ga_tsp.find_fittest_path(graph)
    wall_time = time() - start_wall

    return RunResult(
        seed=seed,
        gpu_enabled=gpu_enabled,
        generations_run=config.max_generations,
        final_fitness=path_cost,  # Store cost as fitness (metric knows it's minimization)
        wall_time=wall_time,
        compute_time=ga_tsp.gpu_compute_time if gpu_enabled else ga_tsp.fitness_compute_time,
        transfer_time=ga_tsp.gpu_transfer_time if gpu_enabled else 0.0,
        fitness_compute_time=ga_tsp.fitness_compute_time,
        extra_data={'graph_size': graph_size, 'final_path_cost': path_cost}
    )


def run_full_tsp_analysis(
    n_runs: int = 30,
    graph_size: int = 50,
    population_size: int = 100,
    max_generations: int = 200,
    tournament_size: int = 5,
    mutation_rate: float = 0.1,
    fit_selection_rate: float = 0.5,
    graph_seed: int = 42,
    verbose: bool = True,
    save: bool = True,
) -> Tuple[Any, List[RunResult], List[RunResult]]:
    config = ExperimentConfig(
        population_size=population_size,
        max_generations=max_generations,
        n_runs=n_runs,
        extra_params={
            'graph_size': graph_size,
            'tournament_size': tournament_size,
            'mutation_rate': mutation_rate,
            'fit_selection_rate': fit_selection_rate,
            'graph_seed': graph_seed,
        }
    )

    output_dir = os.path.join(os.path.dirname(__file__), "logs") if save else None

    return run_full_analysis_generic(
        config=config,
        run_single_func=run_single_tsp,
        fitness_metric=MinimizationMetric("path cost", "final_fitness"),
        problem_name="tsp",
        config_desc=f"TSP experiments (cities={graph_size}, pop={population_size})",
        verbose=verbose,
        save=save,
        output_dir=output_dir,
    )
