from time import time
from typing import List, Tuple, Any
import logging
import os

from genetic.tests.statistical_analysis_base import (
    run_full_analysis_generic,
    ExperimentConfig,
    RunResult,
)
from genetic.tests.fitness_metrics import fitness_maximization
from genetic.tests.null_logger import create_null_logger
from genetic.tests.perceptron.test_xor_perceptron_v5 import GA_PerceptronXOR, PerceptronXOR

logger = logging.getLogger(__name__)


def run_single_perceptron(
    config: ExperimentConfig,
    gpu_enabled: bool,
    seed: int,
) -> RunResult:
    silent_logger = create_null_logger()
    start_wall = time()

    ga = GA_PerceptronXOR(
        logger=silent_logger,
        population_size=config.population_size,
        generations=config.max_generations,
        mutation_rate=config.extra_params.get('mutation_rate', 0.1),
        gene_length=config.extra_params.get('gene_length', 2),
        gpu_enabled=gpu_enabled,
        seed=seed,
    )

    best_weights = ga.run()
    wall_time = time() - start_wall
    perceptron = PerceptronXOR()
    final_fitness = int(perceptron.fitness(best_weights))

    return RunResult(
        seed=seed,
        gpu_enabled=gpu_enabled,
        generations_run=config.max_generations,
        final_fitness=final_fitness,
        wall_time=wall_time,
        compute_time=ga.gpu_compute_time if gpu_enabled else ga.fitness_compute_time,
        transfer_time=ga.gpu_transfer_time if gpu_enabled else 0.0,
        fitness_compute_time=ga.fitness_compute_time,
    )


def run_full_perceptron_analysis(
    n_runs: int = 30,
    population_size: int = 100,
    max_generations: int = 500,
    mutation_rate: float = 0.1,
    verbose: bool = True,
    save: bool = True,
) -> Tuple[Any, List[RunResult], List[RunResult]]:
    config = ExperimentConfig(
        population_size=population_size,
        max_generations=max_generations,
        n_runs=n_runs,
        extra_params={
            'mutation_rate': mutation_rate,
            'gene_length': 2,
        }
    )

    output_dir = os.path.join(os.path.dirname(__file__), "logs") if save else None

    return run_full_analysis_generic(
        config=config,
        run_single_func=run_single_perceptron,
        fitness_metric=fitness_maximization("fitness"),
        problem_name="perceptron",
        config_desc=f"XOR Perceptron experiments (pop={population_size}, gens={max_generations})",
        verbose=verbose,
        save=save,
        output_dir=output_dir,
    )