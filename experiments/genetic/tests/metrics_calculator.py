from dataclasses import dataclass
from typing import List, Protocol, Any
import logging


class RunResultProtocol(Protocol):
    compute_time: float
    transfer_time: float
    wall_time: float


@dataclass
class PerformanceMetrics:
    total_fitness_evals: int
    gpu_compute_avg: float
    cpu_compute_avg: float
    gpu_transfer_avg: float
    gpu_fitness_per_sec: float
    cpu_fitness_per_sec: float
    gpu_overhead_ratio: float
    gpu_overhead_percent: float


def calculate_performance_metrics(
    gpu_results: List[RunResultProtocol],
    cpu_results: List[RunResultProtocol],
    population_size: int,
    max_generations: int,
) -> PerformanceMetrics:
    total_fitness_evals = population_size * max_generations

    gpu_compute_avg = sum(r.compute_time for r in gpu_results) / len(gpu_results)
    cpu_compute_avg = sum(r.compute_time for r in cpu_results) / len(cpu_results)
    gpu_transfer_avg = sum(r.transfer_time for r in gpu_results) / len(gpu_results)

    gpu_fitness_per_sec = total_fitness_evals / gpu_compute_avg if gpu_compute_avg > 0 else 0
    cpu_fitness_per_sec = total_fitness_evals / cpu_compute_avg if cpu_compute_avg > 0 else 0

    gpu_overhead_ratio = gpu_transfer_avg / gpu_compute_avg if gpu_compute_avg > 0 else 0

    # Calculate overhead as percentage of total wall time
    gpu_wall_avg = sum(r.wall_time for r in gpu_results) / len(gpu_results)
    gpu_overhead_percent = (gpu_transfer_avg / gpu_wall_avg * 100) if gpu_wall_avg > 0 else 0

    return PerformanceMetrics(
        total_fitness_evals=total_fitness_evals,
        gpu_compute_avg=gpu_compute_avg,
        cpu_compute_avg=cpu_compute_avg,
        gpu_transfer_avg=gpu_transfer_avg,
        gpu_fitness_per_sec=gpu_fitness_per_sec,
        cpu_fitness_per_sec=cpu_fitness_per_sec,
        gpu_overhead_ratio=gpu_overhead_ratio,
        gpu_overhead_percent=gpu_overhead_percent,
    )


def log_performance_metrics(
    metrics: PerformanceMetrics,
    logger: logging.Logger,
    stats_results: Any,
) -> None:
    logger.info("")
    logger.info("FITNESS FUNCTION PERFORMANCE:")
    logger.info(f"  Total evaluations per run: {metrics.total_fitness_evals:,}")
    logger.info(f"  GPU compute time: {metrics.gpu_compute_avg*1000:.2f}ms ({metrics.gpu_fitness_per_sec:,.0f} evals/sec)")
    logger.info(f"  CPU compute time: {metrics.cpu_compute_avg*1000:.2f}ms ({metrics.cpu_fitness_per_sec:,.0f} evals/sec)")
    logger.info(f"  GPU transfer overhead: {metrics.gpu_transfer_avg*1000:.2f}ms ({metrics.gpu_overhead_percent:.1f}% of total)")


def format_scaling_table_row(
    param_value: Any,
    speedup: float,
    gpu_time_ms: float,
    cpu_time_ms: float,
    gpu_fitness_per_sec: float,
    cpu_fitness_per_sec: float,
) -> str:
    return (
        f"{param_value:<8} "
        f"{speedup:<10.2f} "
        f"{gpu_time_ms:<12.1f} "
        f"{cpu_time_ms:<12.1f} "
        f"{gpu_fitness_per_sec:<12,.0f} "
        f"{cpu_fitness_per_sec:<12,.0f}"
    )
