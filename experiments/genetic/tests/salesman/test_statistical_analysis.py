import pytest

from genetic.tests.logger import CUDA_GA_TestLogger
from genetic.tests.fitness_metrics import MinimizationMetric
from genetic.tests.metrics_calculator import calculate_performance_metrics
from genetic.tests.salesman.statistical_analysis import run_full_tsp_analysis
from genetic.tests.test_statistical_analysis_base import (
    run_full_statistical_analysis,
    run_scaling_analysis,
    run_scaling_summary,
)


CITY_SIZES = [20, 50, 100, 200]
POPULATION_SIZES = [50, 100, 200, 500]


def test_full_statistical_analysis_tsp(logger: CUDA_GA_TestLogger):
    stats, gpu_results, cpu_results = run_full_statistical_analysis(
        run_analysis_func=run_full_tsp_analysis,
        fitness_metric=MinimizationMetric("path cost", "final_fitness"),
        problem_name="tsp",
        test_logger=logger.logger,
        n_runs=50,
        graph_size=50,
        population_size=200,
        max_generations=100,  
        graph_seed=42,  # fixed graph 
    )

    metrics = calculate_performance_metrics(
        gpu_results, cpu_results,
        population_size=200,
        max_generations=100
    )

    logger.end_test("passed", extra_info={
        **stats.to_dict(),
        "total_fitness_evals": metrics.total_fitness_evals,
        "gpu_compute_ms": metrics.gpu_compute_avg * 1000,
        "cpu_compute_ms": metrics.cpu_compute_avg * 1000,
        "gpu_transfer_ms": metrics.gpu_transfer_avg * 1000,
        "problem_type": "tsp",
    })


@pytest.mark.parametrize("graph_size", CITY_SIZES)
def test_tsp_city_scaling(graph_size: int, logger: CUDA_GA_TestLogger):
    results = run_scaling_analysis(
        run_analysis_func=run_full_tsp_analysis,
        fitness_metric=MinimizationMetric("path cost", "final_fitness"),
        problem_name="TSP City",
        test_logger=logger.logger,
        param_name="graph_size",
        param_values=[graph_size],
        n_runs=10,
        population_size=100,
        max_generations=100,   
        graph_seed=42,  # fixed seed per graph_size # TODO: set as a constant
    )

    result = results[0]
    logger.end_test(f"city_scaling_{graph_size}", "passed", extra_info={
        "graph_size": graph_size,
        "speedup": result['stats'].speedup_factor,
        **result,
    })


@pytest.mark.parametrize("population_size", POPULATION_SIZES)
def test_tsp_population_scaling(population_size: int, logger: CUDA_GA_TestLogger):
    results = run_scaling_analysis(
        run_analysis_func=run_full_tsp_analysis,
        fitness_metric=MinimizationMetric("path cost", "final_fitness"),
        problem_name="TSP Population",
        test_logger=logger.logger,
        param_name="population_size",
        param_values=[population_size],
        n_runs=10,
        graph_size=50,  
        max_generations=100,  
        graph_seed=42,  # fixed graph across all population sizes # TODO: set as a constant
    )

    result = results[0]
    logger.end_test(f"population_scaling_{population_size}", "passed", extra_info={
        "population_size": population_size,
        "speedup": result['stats'].speedup_factor,
        **result,
    })


def test_tsp_scaling_summary(logger: CUDA_GA_TestLogger):
    test_logger = logger.logger
    test_logger.info("\n" + "=" * 70)
    test_logger.info("TSP COMPREHENSIVE SCALING SUMMARY")
    test_logger.info("=" * 70)

    logger.start_test("tsp_scaling_summary")

    # City size scaling
    city_results = run_scaling_analysis(
        run_analysis_func=run_full_tsp_analysis,
        fitness_metric=MinimizationMetric("path cost", "final_fitness"),
        problem_name="TSP City",
        test_logger=test_logger,
        param_name="graph_size",
        param_values=CITY_SIZES,
        n_runs=5,
        population_size=100,
        max_generations=100,  
        graph_seed=42,  # fixed seed per graph_size # TODO: set as a constant
    )

    run_scaling_summary(
        results=city_results,
        problem_name="TSP City",
        param_name="graph_size",
        test_logger=test_logger,
        **{"Scaling": "O(n²) due to distance matrix"}
    )

    # Population scaling
    pop_results = run_scaling_analysis(
        run_analysis_func=run_full_tsp_analysis,
        fitness_metric=MinimizationMetric("path cost", "final_fitness"),
        problem_name="TSP Population",
        test_logger=test_logger,
        param_name="population_size",
        param_values=POPULATION_SIZES,
        n_runs=5,
        graph_size=50,  
        max_generations=100,  
        graph_seed=42,  # fixed graph for all population sizes # TODO: set as a constant
    )

    run_scaling_summary(
        results=pop_results,
        problem_name="TSP Population",
        param_name="population_size",
        test_logger=test_logger,
        **{"GPU": "O(1)", "CPU": "O(n)"}
    )

    logger.end_test("passed", extra_info={
        "city_scaling_results": [
            {
                "graph_size": r['graph_size'],
                "speedup": r['stats'].speedup_factor,
                "gpu_time_ms": r['stats'].gpu_mean_time * 1000,
                "cpu_time_ms": r['stats'].cpu_mean_time * 1000,
            } for r in city_results
        ],
        "population_scaling_results": [
            {
                "population_size": r['population_size'],
                "speedup": r['stats'].speedup_factor,
                "gpu_time_ms": r['stats'].gpu_mean_time * 1000,
                "cpu_time_ms": r['stats'].cpu_mean_time * 1000,
            } for r in pop_results
        ],
    })
