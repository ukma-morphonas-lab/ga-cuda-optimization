# pyright: reportCallIssue=false
import pytest

from genetic.tests.logger import CUDA_GA_TestLogger
from genetic.tests.fitness_metrics import fitness_maximization
from genetic.tests.metrics_calculator import calculate_performance_metrics, log_performance_metrics
from genetic.tests.test_statistical_analysis_base import (
    run_full_statistical_analysis,
    run_scaling_analysis,
    run_scaling_summary,
)
from genetic.tests.perceptron.statistical_analysis import run_full_perceptron_analysis


POPULATION_SIZES = [50, 100, 500, 1000]


def test_full_statistical_analysis(logger: CUDA_GA_TestLogger):
    """Full statistical analysis (30 runs) - scientifically valid"""
    stats, gpu_results, cpu_results = run_full_statistical_analysis(
        run_analysis_func=run_full_perceptron_analysis,
        fitness_metric=fitness_maximization("fitness"),
        problem_name="perceptron",
        test_logger=logger.logger,
        n_runs=30,
        population_size=100,
        max_generations=500,
    )

    assert stats.n_runs >= 30

    metrics = calculate_performance_metrics(
        gpu_results, cpu_results,
        population_size=100,
        max_generations=500
    )

    logger.end_test("passed", extra_info={
        **stats.to_dict(),
        "total_fitness_evals": metrics.total_fitness_evals,
        "gpu_compute_ms": metrics.gpu_compute_avg * 1000,
        "cpu_compute_ms": metrics.cpu_compute_avg * 1000,
        "gpu_transfer_ms": metrics.gpu_transfer_avg * 1000,
        "gpu_fitness_per_sec": metrics.gpu_fitness_per_sec,
        "cpu_fitness_per_sec": metrics.cpu_fitness_per_sec,
    })


@pytest.mark.parametrize("population_size", POPULATION_SIZES)
def test_scaling_analysis(population_size: int, logger: CUDA_GA_TestLogger):
    """Test scaling behavior across different population sizes"""
    results = run_scaling_analysis(
        run_analysis_func=run_full_perceptron_analysis,
        fitness_metric=fitness_maximization("fitness"),
        problem_name="Perceptron",
        test_logger=logger.logger,
        param_name="population_size",
        param_values=[population_size],
        n_runs=10,
        max_generations=200,
    )

    result = results[0]
    logger.end_test("passed", extra_info={
        "population_size": population_size,
        "speedup": result['stats'].speedup_factor,
        **result,
    })


def test_scaling_summary(logger: CUDA_GA_TestLogger):
    """Generate comprehensive scaling summary"""
    results = run_scaling_analysis(
        run_analysis_func=run_full_perceptron_analysis,
        fitness_metric=fitness_maximization("fitness"),
        problem_name="Perceptron",
        test_logger=logger.logger,
        param_name="population_size",
        param_values=POPULATION_SIZES,
        n_runs=5,
        max_generations=200,
    )

    run_scaling_summary(
        results=results,
        problem_name="Perceptron",
        param_name="population_size",
        test_logger=logger.logger,
        **{"Scaling": "GPU scales O(1), CPU scales O(n)"}
    )

    logger.end_test("passed", extra_info={"results": results})
