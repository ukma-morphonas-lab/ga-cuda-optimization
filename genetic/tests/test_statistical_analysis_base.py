import pytest
from typing import Callable, Any, Dict, List, Tuple

from genetic.tests.metrics_calculator import calculate_performance_metrics, log_performance_metrics
from genetic.tests.fitness_metrics import FitnessMetric


def run_full_statistical_analysis(
    run_analysis_func: Callable[..., Tuple[Any, List[Any], List[Any]]],
    fitness_metric: FitnessMetric,
    problem_name: str,
    test_logger: Any,
    n_runs: int = 30,
    **kwargs
):
    test_logger.info(f"\n{'='*80}")
    test_logger.info(f"FULL {problem_name.upper()} STATISTICAL ANALYSIS ({n_runs} runs)")
    test_logger.info("=" * 80)

    stats, gpu_results, cpu_results = run_analysis_func(
        n_runs=n_runs,
        verbose=True,
        save=True,
        **kwargs
    )

    assert stats.n_runs >= n_runs

    # Calculate performance metrics using helper
    population_size = kwargs.get('population_size', 100)
    max_generations = kwargs.get('max_generations', 200)
    metrics = calculate_performance_metrics(
        gpu_results, cpu_results, population_size, max_generations
    )

    # Log results
    test_logger.info(f"GPU Mean {fitness_metric.get_name()}: {stats.gpu_mean_fitness:.2f}")
    test_logger.info(f"CPU Mean {fitness_metric.get_name()}: {stats.cpu_mean_fitness:.2f}")
    test_logger.info(f"Speedup: {stats.speedup_factor:.2f}x")
    test_logger.info(f"Time p-value: {stats.p_value_time:.4f}")

    log_performance_metrics(metrics, test_logger, stats)

    if stats.time_significant:
        test_logger.info(f"SIGNIFICANT: Time difference is statistically significant (p={stats.p_value_time:.4f})")
    else:
        test_logger.info(f"NOT SIGNIFICANT: Time difference is NOT statistically significant (p={stats.p_value_time:.4f})")

    return stats, gpu_results, cpu_results


def run_scaling_analysis(
    run_analysis_func: Callable[..., Tuple[Any, List[Any], List[Any]]],
    fitness_metric: FitnessMetric,
    problem_name: str,
    test_logger: Any,
    param_name: str,
    param_values: List[Any],
    n_runs: int = 10,
    **fixed_kwargs
):
    test_logger.info(f"\n{'='*60}")
    test_logger.info(f"{problem_name.upper()} SCALING ANALYSIS: {param_name}")
    test_logger.info("=" * 60)

    results = []

    for param_value in param_values:
        test_logger.info(f"\n--- Testing {param_name} = {param_value} ---")

        # Create kwargs for this run
        run_kwargs = fixed_kwargs.copy()
        run_kwargs[param_name] = param_value
        run_kwargs['n_runs'] = n_runs
        run_kwargs['verbose'] = False
        run_kwargs['save'] = True

        stats, gpu_results, cpu_results = run_analysis_func(**run_kwargs)

        # Calculate metrics using helper
        pop_size = run_kwargs.get('population_size', run_kwargs.get('graph_size', 100))
        max_gens = run_kwargs.get('max_generations', 200)
        metrics = calculate_performance_metrics(
            gpu_results, cpu_results, pop_size, max_gens
        )

        # Log results
        test_logger.info(f"{param_name} {param_value}: Speedup = {stats.speedup_factor:.2f}x")
        test_logger.info(f"  Wall time:  GPU={stats.gpu_mean_time*1000:.2f}ms, CPU={stats.cpu_mean_time*1000:.2f}ms")
        test_logger.info(f"  Compute:    GPU={metrics.gpu_compute_avg*1000:.2f}ms, CPU={metrics.cpu_compute_avg*1000:.2f}ms")
        test_logger.info(f"  Transfer:   GPU={metrics.gpu_transfer_avg*1000:.2f}ms")
        test_logger.info(f"  Fitness evaluations: {metrics.total_fitness_evals:,}")
        test_logger.info(f"  Fitness/sec: GPU={metrics.gpu_fitness_per_sec:,.0f}, CPU={metrics.cpu_fitness_per_sec:,.0f}")
        test_logger.info(f"  GPU overhead ratio: {metrics.gpu_overhead_ratio:.2%}")

        results.append({
            param_name: param_value,
            'stats': stats,
            'gpu_results': gpu_results,
            'cpu_results': cpu_results,
            'gpu_compute_avg': metrics.gpu_compute_avg,
            'cpu_compute_avg': metrics.cpu_compute_avg,
            'gpu_transfer_avg': metrics.gpu_transfer_avg,
            'total_fitness_evals': metrics.total_fitness_evals,
            'gpu_fitness_per_sec': metrics.gpu_fitness_per_sec,
            'cpu_fitness_per_sec': metrics.cpu_fitness_per_sec,
            'gpu_overhead_ratio': metrics.gpu_overhead_ratio,
        })

    return results


def run_scaling_summary(
    results: List[Dict],
    problem_name: str,
    param_name: str,
    test_logger: Any,
    **summary_kwargs
):
    test_logger.info(f"\n{'='*70}")
    test_logger.info(f"{problem_name.upper()} FITNESS FUNCTION SCALING SUMMARY")
    test_logger.info("=" * 70)

    # Extract data for summary table
    param_values = [r[param_name] for r in results]
    speedups = [r['stats'].speedup_factor for r in results]
    gpu_times = [r['stats'].gpu_mean_time * 1000 for r in results]
    cpu_times = [r['stats'].cpu_mean_time * 1000 for r in results]
    gpu_fitness_per_sec = [r['gpu_fitness_per_sec'] for r in results]
    cpu_fitness_per_sec = [r['cpu_fitness_per_sec'] for r in results]

    # Print scaling table
    test_logger.info("")
    test_logger.info(f"{'Param':<8} {'Speedup':<10} {'GPU (ms)':<12} {'CPU (ms)':<12} {'GPU fit/s':<12} {'CPU fit/s':<12}")
    test_logger.info("-" * 70)

    for i, param_val in enumerate(param_values):
        test_logger.info(
            f"{param_val:<8} "
            f"{speedups[i]:<10.2f} "
            f"{gpu_times[i]:<12.1f} "
            f"{cpu_times[i]:<12.1f} "
            f"{gpu_fitness_per_sec[i]:<12,.0f} "
            f"{cpu_fitness_per_sec[i]:<12,.0f}"
        )

    test_logger.info("-" * 70)

    # Calculate scaling efficiency if we have multiple results
    if len(results) >= 2:
        first = results[0]
        last = results[-1]

        param_ratio = last[param_name] / first[param_name] if first[param_name] != 0 else 0
        speedup_ratio = last['stats'].speedup_factor / first['stats'].speedup_factor if first['stats'].speedup_factor > 0 else 0

        test_logger.info("")
        test_logger.info("SCALING ANALYSIS:")
        test_logger.info(f"  {param_name} increase: {first[param_name]} → {last[param_name]} ({param_ratio:.0f}x)")

        if speedup_ratio > 0:
            test_logger.info(f"  Speedup change: {first['stats'].speedup_factor:.2f}x → {last['stats'].speedup_factor:.2f}x ({speedup_ratio:.1f}x improvement)")

        # Add problem-specific scaling information
        for key, value in summary_kwargs.items():
            test_logger.info(f"  {key}: {value}")

        # Breakeven analysis
        breakeven_param = None
        for r in results:
            if r['stats'].speedup_factor >= 1.0:
                breakeven_param = r[param_name]
                break

        if breakeven_param:
            test_logger.info(f"  GPU breakeven point: {param_name} ≥ {breakeven_param}")
        else:
            test_logger.info("  GPU breakeven point: not reached in tested range")

    test_logger.info("=" * 70)

    return results


@pytest.fixture
def quick_config_common() -> dict:
    # fast testing
    return {
        'population_size': 50,
        'max_generations': 100,
        'n_runs': 5,
    }


@pytest.fixture
def valid_config_common() -> dict:
    # statistically valid configuration
    return {
        'population_size': 100,
        'max_generations': 500,
        'n_runs': 30,
    }
