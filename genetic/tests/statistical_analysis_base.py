
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple, Callable, Protocol
import json
import os
from datetime import datetime
import logging
import numpy as np
from scipy import stats
from scipy.stats import norm

from genetic.tests.constants import RANDOM_SEED
from genetic.tests.fitness_metrics import FitnessMetric

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResults:
    n_runs: int

    # fitness (problem-specific metric)
    gpu_mean_fitness: float
    cpu_mean_fitness: float
    gpu_std_fitness: float
    cpu_std_fitness: float
    p_value_fitness: float  # wilcoxon signed-rank
    fitness_difference: float  # gpu - cpu

    # wall-clock time
    gpu_mean_time: float
    cpu_mean_time: float
    gpu_std_time: float
    cpu_std_time: float
    p_value_time: float  # paired t-test
    speedup_factor: float
    speedup_ci_lower: float
    speedup_ci_upper: float

    # significance flags
    time_significant: bool
    fitness_significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self, fitness_name: str = "fitness", fitness_goal: str = "higher") -> str:
        diff_desc = "(GPU - CPU)"

        lines = [
            "=" * 70,
            "STATISTICAL ANALYSIS RESULTS",
            "=" * 70,
            f"Number of runs: {self.n_runs}",
            "",
            fitness_name.upper(),
            "-" * 40,
            f"  GPU: {self.gpu_mean_fitness:.2f} ± {self.gpu_std_fitness:.2f}",
            f"  CPU: {self.cpu_mean_fitness:.2f} ± {self.cpu_std_fitness:.2f}",
            f"  Difference {diff_desc}: {self.fitness_difference:.2f}",
            f"  Wilcoxon p-value: {self.p_value_fitness:.4f}",
            f"  Significant (α=0.05): {'YES' if self.fitness_significant else 'NO'}",
            "",
            "WALL-CLOCK TIME",
            "-" * 40,
            f"  GPU: {self.gpu_mean_time*1000:.2f} ± {self.gpu_std_time*1000:.2f} ms",
            f"  CPU: {self.cpu_mean_time*1000:.2f} ± {self.cpu_std_time*1000:.2f} ms",
            f"  Speedup: {self.speedup_factor:.2f}x",
            f"  Speedup 95% CI: [{self.speedup_ci_lower:.2f}x, {self.speedup_ci_upper:.2f}x]",
            f"  Paired t-test p-value: {self.p_value_time:.4f}",
            f"  Significant (α=0.05): {'YES' if self.time_significant else 'NO'}",
            "=" * 70,
        ]
        return "\n".join(lines)


def _bootstrap_speedup_ci(
    gpu_times: np.ndarray,
    cpu_times: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> Tuple[float, float]:
    n = len(gpu_times)
    rng = np.random.RandomState(RANDOM_SEED)  # fixed seed for reproducibility

    speedup_samples = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # resample with replacement (maintaining pairing)
        indices = rng.choice(n, size=n, replace=True)
        gpu_sample = gpu_times[indices]
        cpu_sample = cpu_times[indices]

        # calculate speedup for this bootstrap sample
        mean_gpu_sample = np.mean(gpu_sample)
        if mean_gpu_sample > 0:
            speedup_samples[i] = np.mean(cpu_sample) / mean_gpu_sample
        else:
            speedup_samples[i] = 1.0

    # percentile-based confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(speedup_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(speedup_samples, 100 * (1 - alpha / 2)))

    return ci_lower, ci_upper


def analyze_results(
    gpu_fitness_values: List[float],
    cpu_fitness_values: List[float],
    gpu_times: List[float],
    cpu_times: List[float],
    alpha: float = 0.05,
) -> StatisticalResults:
    """
    Perform statistical analysis on paired GPU vs CPU results.

    - Wilcoxon signed-rank test for fitness comparison (paired, non-parametric)
    - Paired t-test for wall-clock time comparison
    - Bootstrap CI for speedup ratio
    """
    n = len(gpu_fitness_values)

    # Extract metrics
    gpu_fitness = np.array(gpu_fitness_values)
    cpu_fitness = np.array(cpu_fitness_values)
    gpu_time_array = np.array(gpu_times)
    cpu_time_array = np.array(cpu_times)

    # Wilcoxon signed-rank test for fitness (paired, non-parametric)
    fitness_diff = gpu_fitness - cpu_fitness
    p_wilcoxon: float
    if np.all(fitness_diff == 0):
        p_wilcoxon = 1.0
    else:
        try:
            wilcox_result = stats.wilcoxon(gpu_fitness, cpu_fitness, zero_method='wilcox')
            p_wilcoxon = float(wilcox_result.pvalue)  # type: ignore
        except ValueError:
            p_wilcoxon = 1.0

    # Paired t-test for wall-clock time
    ttest_result = stats.ttest_rel(gpu_time_array, cpu_time_array)
    p_ttest: float = float(ttest_result.pvalue)

    # Speedup calculation with bootstrap CI
    mean_gpu = float(np.mean(gpu_time_array))
    mean_cpu = float(np.mean(cpu_time_array))
    speedup_factor = mean_cpu / mean_gpu if mean_gpu > 0 else 1.0

    # bootstrap CI for speedup
    if len(gpu_time_array) > 1:
        speedup_ci_lower, speedup_ci_upper = _bootstrap_speedup_ci(
            gpu_time_array, cpu_time_array, confidence=0.95, n_bootstrap=10000
        )
    else:
        speedup_ci_lower, speedup_ci_upper = speedup_factor, speedup_factor

    return StatisticalResults(
        n_runs=n,
        gpu_mean_fitness=float(np.mean(gpu_fitness)),
        cpu_mean_fitness=float(np.mean(cpu_fitness)),
        gpu_std_fitness=float(np.std(gpu_fitness, ddof=1)) if n > 1 else 0.0,
        cpu_std_fitness=float(np.std(cpu_fitness, ddof=1)) if n > 1 else 0.0,
        p_value_fitness=float(p_wilcoxon),
        fitness_difference=float(np.mean(fitness_diff)),
        gpu_mean_time=float(np.mean(gpu_time_array)),
        cpu_mean_time=float(np.mean(cpu_time_array)),
        gpu_std_time=float(np.std(gpu_time_array, ddof=1)) if n > 1 else 0.0,
        cpu_std_time=float(np.std(cpu_time_array, ddof=1)) if n > 1 else 0.0,
        p_value_time=float(p_ttest),
        speedup_factor=speedup_factor,
        speedup_ci_lower=speedup_ci_lower,
        speedup_ci_upper=speedup_ci_upper,
        time_significant=p_ttest < alpha,
        fitness_significant=p_wilcoxon < alpha,
    )


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    z_alpha = norm.ppf(1 - alpha / 2)  # two-tailed
    z_beta = norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))


def save_results(
    config: Dict[str, Any],
    gpu_results: List[Dict[str, Any]],
    cpu_results: List[Dict[str, Any]],
    stats_results: StatisticalResults,
    output_dir: str,
    problem_name: str = "generic",
) -> str:
    """
    Save statistical analysis results to JSON.

    Args:
        output_dir: Directory to save results (must be provided by caller)
        problem_name: Used only for filename, not directory resolution
    """
    # Look for the most recent run directory (created by logger)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.path.exists(output_dir):
        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_") and os.path.isdir(os.path.join(output_dir, d))]
        if run_dirs:
            # Sort by modification time, get the most recent
            run_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
            run_dir = os.path.join(output_dir, run_dirs[0])
        else:
            # No run directories found, create a new one
            run_dir = os.path.join(output_dir, f"run_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)
    else:
        # Output dir doesn't exist, create it and a run dir
        os.makedirs(output_dir, exist_ok=True)
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

    filename = os.path.join(
        run_dir,
        f"statistical_analysis_{problem_name}_{stats_results.n_runs}runs_{timestamp}.json"
    )

    data = {
        "timestamp": timestamp,
        "config": config,
        "statistical_results": stats_results.to_dict(),
        "gpu_results": gpu_results,
        "cpu_results": cpu_results,
        "power_analysis": {
            "small_effect": power_analysis(0.2),
            "medium_effect": power_analysis(0.5),
            "large_effect": power_analysis(0.8),
        },
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return filename


def print_power_analysis_table():
    logger.info("POWER ANALYSIS REFERENCE TABLE")
    logger.info("(α=0.05, power=0.80, two-sample t-test)")
    logger.info("=" * 50)
    logger.info(f"{'Effect Size':<20} {'Required N':<12}")
    logger.info("-" * 50)
    logger.info(f"{'Small (0.2)':<20} {power_analysis(0.2):<12}")
    logger.info(f"{'Medium (0.5)':<20} {power_analysis(0.5):<12}")
    logger.info(f"{'Large (0.8)':<20} {power_analysis(0.8):<12}")
    logger.info("=" * 50)


class GAExperimentProtocol(Protocol):
    def to_dict(self) -> Dict[str, Any]: ...


class GAResultProtocol(Protocol):
    seed: int
    gpu_enabled: bool
    generations_run: int
    wall_time: float
    compute_time: float
    transfer_time: float

    def to_dict(self) -> Dict[str, Any]: ...


@dataclass
class ExperimentConfig:
    """Unified configuration for all GA experiments."""
    population_size: int = 100
    max_generations: int = 200
    n_runs: int = 30
    alpha: float = 0.05
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        # Flatten extra_params into main dict
        extra = base.pop('extra_params', {})
        base.update(extra)
        return base


@dataclass
class RunResult:
    """Unified result structure for all GA experiments."""
    seed: int
    gpu_enabled: bool
    generations_run: int
    final_fitness: float  # Universal name (use fitness_metric to interpret)
    wall_time: float
    compute_time: float
    transfer_time: float
    fitness_compute_time: float = 0.0
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        extra = base.pop('extra_data', {})
        base.update(extra)
        return base


def run_experiment_generic(
    config: ExperimentConfig,
    run_single_func: Callable[[Any, bool, int], Any],
    fitness_metric: FitnessMetric,
    verbose: bool = True,
    config_desc: str = "generic experiment",
) -> Tuple[List[Any], List[Any]]:
    gpu_results: List[Any] = []
    cpu_results: List[Any] = []

    if verbose:
        logger.info(f"Running {config.n_runs} paired {config_desc}...")
        logger.info("-" * 50)

    for i in range(config.n_runs):
        seed = i

        # run GPU experiment
        gpu_result = run_single_func(config, gpu_enabled=True, seed=seed)
        gpu_results.append(gpu_result)

        # run CPU experiment with same seed
        cpu_result = run_single_func(config, gpu_enabled=False, seed=seed)
        cpu_results.append(cpu_result)

        if verbose:
            gpu_fitness = fitness_metric.extract_value(gpu_result)
            cpu_fitness = fitness_metric.extract_value(cpu_result)
            logger.info(f"  Run {i+1:3d}/{config.n_runs}: "
                  f"GPU fitness={gpu_fitness:8.2f}, {gpu_result.wall_time*1000:7.2f}ms | "
                  f"CPU fitness={cpu_fitness:8.2f}, {cpu_result.wall_time*1000:7.2f}ms")

    return gpu_results, cpu_results


def run_full_analysis_generic(
    config: Any,
    run_single_func: Callable[[Any, bool, int], Any],
    fitness_metric: FitnessMetric,
    problem_name: str = "generic",
    config_desc: str = "generic experiment",
    verbose: bool = True,
    save: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[StatisticalResults, List[Any], List[Any]]:
    gpu_results, cpu_results = run_experiment_generic(
        config, run_single_func, fitness_metric, verbose, config_desc
    )

    # extract fitness and time data
    gpu_fitness = []
    cpu_fitness = []
    gpu_times = []
    cpu_times = []

    for gpu_r, cpu_r in zip(gpu_results, cpu_results):
        # Use fitness metric to extract values
        gpu_fit = fitness_metric.extract_value(gpu_r)
        cpu_fit = fitness_metric.extract_value(cpu_r)

        gpu_fitness.append(float(gpu_fit))
        cpu_fitness.append(float(cpu_fit))
        gpu_times.append(gpu_r.wall_time)
        cpu_times.append(cpu_r.wall_time)

    stats_results = analyze_results(gpu_fitness, cpu_fitness, gpu_times, cpu_times, alpha=config.alpha)

    if verbose:
        logger.info(stats_results.summary(
            fitness_name=fitness_metric.get_name(),
            fitness_goal=fitness_metric.get_goal_description()
        ))

    if save:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save=True")
        gpu_dicts = [r.to_dict() for r in gpu_results]
        cpu_dicts = [r.to_dict() for r in cpu_results]
        filepath = save_results(config.to_dict(), gpu_dicts, cpu_dicts, stats_results,
                              output_dir=output_dir, problem_name=problem_name)
        if verbose:
            logger.info(f"Results saved to: {filepath}")

    return stats_results, gpu_results, cpu_results
