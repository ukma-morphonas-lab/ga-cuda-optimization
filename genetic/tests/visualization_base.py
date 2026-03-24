from pathlib import Path
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib style — academic minimalist
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.5,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Latin Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'legend.frameon': True,
    'legend.edgecolor': '#CCCCCC',
    'legend.framealpha': 0.9,
    'legend.fontsize': 9,
})

# Unified muted academic palette
GPU_COLOR = '#4C72B0'      # Steel blue
CPU_COLOR = '#C44E52'      # Muted coral
TRANSFER_COLOR = '#DD8452'  # Warm amber
COMPUTE_COLOR = '#55A868'   # Muted teal
ACCENT_COLOR = '#8172B3'    # Muted purple
BAR_EDGE = 'none'


class GAVisualizer(ABC):
    def __init__(self, problem_name: str, logs_dir: Path, output_dir: Path):
        self.problem_name = problem_name
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    @abstractmethod
    def load_scaling_results(self, param_name: str, param_values: List[Any]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def load_full_analysis(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_scaling_parameters(self) -> List[Tuple[str, List[Any], str]]:
        pass

    def plot_speedup_vs_parameter(self, results: List[Dict], param_name: str, param_display: str, output_dir: Path):
        fig, ax = plt.subplots(figsize=(10, 6))

        param_values = [r['config'][param_name] for r in results]
        speedups = [r['statistical_results']['speedup_factor'] for r in results]

        # Bar chart
        bars = ax.bar(range(len(param_values)), speedups, color=GPU_COLOR, edgecolor=BAR_EDGE)

        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            label = f'{speedup:.2f}x'
            color = '#333333'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=12, color=color)

        # Reference line at 1x (breakeven)
        ax.axhline(y=1, color='#888888', linestyle='--', linewidth=1.2, label='Breakeven (1x)')

        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([str(p) for p in param_values])
        ax.set_xlabel(param_display)
        ax.set_ylabel('Speedup Factor (CPU time / GPU time)')
        ax.set_title(f'GPU Speedup vs {param_display}\n(Higher = GPU faster)')
        ax.legend(loc='upper left')
        ax.set_ylim(0, max(speedups) * 1.2)

        plt.tight_layout()
        plt.savefig(output_dir / f'{self.problem_name.lower()}_speedup_vs_{param_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.problem_name.lower()}_speedup_vs_{param_name}.png")


    def plot_time_comparison(self, results: List[Dict], param_name: str, param_display: str, output_dir: Path):
        fig, ax = plt.subplots(figsize=(12, 6))

        param_values = [r['config'][param_name] for r in results]
        gpu_times = [r['statistical_results']['gpu_mean_time'] * 1000 for r in results]
        cpu_times = [r['statistical_results']['cpu_mean_time'] * 1000 for r in results]

        x = np.arange(len(param_values))
        width = 0.35

        bars_gpu = ax.bar(x - width/2, gpu_times, width, label='GPU', color=GPU_COLOR, edgecolor=BAR_EDGE)
        bars_cpu = ax.bar(x + width/2, cpu_times, width, label='CPU', color=CPU_COLOR, edgecolor=BAR_EDGE)

        # Add value labels
        for bar in bars_gpu:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        for bar in bars_cpu:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in param_values])
        ax.set_xlabel(param_display)
        ax.set_ylabel('Wall Time (ms)')
        ax.set_title(f'GPU vs CPU Execution Time by {param_display}')
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_dir / f'{self.problem_name.lower()}_time_comparison_{param_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.problem_name.lower()}_time_comparison_{param_name}.png")


    def plot_gpu_time_breakdown(self, results: List[Dict], param_name: str, param_display: str, output_dir: Path):
        fig, ax = plt.subplots(figsize=(10, 6))

        param_values = []
        compute_times = []
        transfer_times = []

        for r in results:
            param_values.append(r['config'][param_name])
            gpu_results = r['gpu_results']
            avg_compute = np.mean([gr['compute_time'] for gr in gpu_results]) * 1000
            avg_transfer = np.mean([gr['transfer_time'] for gr in gpu_results]) * 1000
            compute_times.append(avg_compute)
            transfer_times.append(avg_transfer)

        x = np.arange(len(param_values))
        width = 0.6

        bars_compute = ax.bar(x, compute_times, width, label='GPU Compute', color=COMPUTE_COLOR, edgecolor=BAR_EDGE)
        bars_transfer = ax.bar(x, transfer_times, width, bottom=compute_times, label='GPU Transfer', color=TRANSFER_COLOR, edgecolor=BAR_EDGE)

        # Add total time labels
        for i, (comp, trans) in enumerate(zip(compute_times, transfer_times)):
            total = comp + trans
            overhead = trans / total * 100 if total > 0 else 0
            ax.annotate('.0f',
                        xy=(i, total),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in param_values])
        ax.set_xlabel(param_display)
        ax.set_ylabel('Time (ms)')
        ax.set_title('GPU Time Breakdown: Compute vs Memory Transfer')
        ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(output_dir / f'{self.problem_name.lower()}_gpu_time_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.problem_name.lower()}_gpu_time_breakdown.png")


    def plot_fitness_throughput(self, results: List[Dict], param_name: str, param_display: str, output_dir: Path):
        fig, ax = plt.subplots(figsize=(10, 6))

        param_values = []
        gpu_throughput = []
        cpu_throughput = []

        for r in results:
            param_val = r['config'][param_name]
            pop_size = r['config']['population_size']
            max_gens = r['config']['max_generations']
            total_evals = pop_size * max_gens

            gpu_results = r['gpu_results']
            cpu_results = r['cpu_results']

            avg_gpu_compute = np.mean([gr['compute_time'] for gr in gpu_results])
            avg_cpu_compute = np.mean([cr['compute_time'] for cr in cpu_results])

            param_values.append(param_val)
            gpu_throughput.append(total_evals / avg_gpu_compute / 1000)  # thousands per sec
            cpu_throughput.append(total_evals / avg_cpu_compute / 1000)

        x = np.arange(len(param_values))
        width = 0.35

        bars_gpu = ax.bar(x - width/2, gpu_throughput, width, label='GPU', color=GPU_COLOR, edgecolor=BAR_EDGE)
        bars_cpu = ax.bar(x + width/2, cpu_throughput, width, label='CPU', color=CPU_COLOR, edgecolor=BAR_EDGE)

        # Add value labels
        for bar in bars_gpu:
            height = bar.get_height()
            ax.annotate('.0f',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        for bar in bars_cpu:
            height = bar.get_height()
            ax.annotate('.0f',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in param_values])
        ax.set_xlabel(param_display)
        ax.set_ylabel('Fitness Evaluations (thousands/sec)')
        ax.set_title('Fitness Function Throughput: GPU vs CPU')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f'{self.problem_name.lower()}_fitness_throughput.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.problem_name.lower()}_fitness_throughput.png")


    def plot_scaling_efficiency(self, results: List[Dict], param_name: str, param_display: str,
                               complexity: str, output_dir: Path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        param_values = [r['config'][param_name] for r in results]
        gpu_times = [r['statistical_results']['gpu_mean_time'] * 1000 for r in results]
        cpu_times = [r['statistical_results']['cpu_mean_time'] * 1000 for r in results]

        # Left plot: Linear scale
        ax1.plot(param_values, gpu_times, 'o-', color=GPU_COLOR, linewidth=1.5, markersize=6, label='GPU')
        ax1.plot(param_values, cpu_times, 's-', color=CPU_COLOR, linewidth=1.5, markersize=6, label='CPU')

        ax1.set_xlabel(param_display)
        ax1.set_ylabel('Wall Time (ms)')
        ax1.set_title('Execution Time Scaling (Linear)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Right plot: Log-log scale to show scaling behavior
        ax2.loglog(param_values, gpu_times, 'o-', color=GPU_COLOR, linewidth=1.5, markersize=6, label=f'GPU ~ {complexity}')
        ax2.loglog(param_values, cpu_times, 's-', color=CPU_COLOR, linewidth=1.5, markersize=6, label=f'CPU ~ {complexity}')

        # Add reference lines
        x_ref = np.array(param_values)
        if complexity == "O(n)":
            y_ref = gpu_times[0] * (x_ref / x_ref[0])
            ax2.loglog(x_ref, y_ref, '--', color=GPU_COLOR, alpha=0.5, linewidth=1, label='O(n) reference')
        elif complexity == "O(n²)":
            y_ref = gpu_times[0] * (x_ref / x_ref[0]) ** 2
            ax2.loglog(x_ref, y_ref, '--', color=GPU_COLOR, alpha=0.5, linewidth=1, label='O(n²) reference')

        ax2.set_xlabel(param_display)
        ax2.set_ylabel('Wall Time (ms)')
        ax2.set_title('Execution Time Scaling (Log-Log)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{self.problem_name.lower()}_scaling_efficiency.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.problem_name.lower()}_scaling_efficiency.png")


    def plot_statistical_significance(self, full_analysis: Dict, output_dir: Path):
        if not full_analysis:
            print("No full analysis data found, skipping statistical significance plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        gpu_results = full_analysis['gpu_results']
        cpu_results = full_analysis['cpu_results']
        stats = full_analysis['statistical_results']

        n_runs = len(gpu_results)

        # Left plot: Run-by-run times
        runs = range(1, n_runs + 1)
        gpu_times = [r['wall_time'] * 1000 for r in gpu_results]
        cpu_times = [r['wall_time'] * 1000 for r in cpu_results]

        ax1.scatter(runs, gpu_times, color=GPU_COLOR, s=30, alpha=0.7, label='GPU', edgecolors='none')
        ax1.scatter(runs, cpu_times, color=CPU_COLOR, s=30, alpha=0.7, label='CPU', edgecolors='none')

        # Mean lines
        ax1.axhline(y=np.mean(gpu_times), color=GPU_COLOR, linestyle='--', linewidth=1.2, alpha=0.8)
        ax1.axhline(y=np.mean(cpu_times), color=CPU_COLOR, linestyle='--', linewidth=1.2, alpha=0.8)

        ax1.set_xlabel('Run Number')
        ax1.set_ylabel('Wall Time (ms)')
        ax1.set_title(f'Run-by-Run Comparison (N={n_runs})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Right plot: Box plot with stats
        box_data = [gpu_times, cpu_times]
        bp = ax2.boxplot(box_data, tick_labels=['GPU', 'CPU'], patch_artist=True)

        bp['boxes'][0].set_facecolor(GPU_COLOR)
        bp['boxes'][1].set_facecolor(CPU_COLOR)

        for box in bp['boxes']:
            box.set_alpha(0.7)

        ax2.set_ylabel('Wall Time (ms)')
        ax2.set_title('Distribution Comparison')
        ax2.set_yscale('log')

        # Add statistical annotation
        stats_text = (
            f"N = {n_runs}\n"
            f"Speedup: {stats['speedup_factor']:.2f}x\n"
            f"p-value: {stats['p_value_time']:.2e}\n"
            f"Significant: {'Yes' if stats['time_significant'] else 'No'}"
        )
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCCCCC', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_dir / f'{self.problem_name.lower()}_statistical_significance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.problem_name.lower()}_statistical_significance.png")


    def generate_all_plots(self):
        print(f"Loading {self.problem_name} results from: {self.logs_dir}")
        print(f"Saving charts to: {self.output_dir}")
        print("-" * 50)

        scaling_params = self.get_scaling_parameters()
        full_analysis = self.load_full_analysis()

        # Generate plots for each scaling parameter
        for param_name, param_values, param_display in scaling_params:
            results = self.load_scaling_results(param_name, param_values)

            if results:
                print(f"Found {len(results)} {param_display} configurations")
                for r in results:
                    speedup = r['statistical_results']['speedup_factor']
                    param_val = r['config'][param_name]
                    print(f"  - {param_name}={param_val}, speedup={speedup:.2f}x")

                # Generate plots
                self.plot_speedup_vs_parameter(results, param_name, param_display, self.output_dir)
                self.plot_time_comparison(results, param_name, param_display, self.output_dir)
                self.plot_gpu_time_breakdown(results, param_name, param_display, self.output_dir)
                self.plot_fitness_throughput(results, param_name, param_display, self.output_dir)

                # Scaling efficiency with problem-specific complexity
                complexity = "O(n)" if param_name == "population_size" else "O(n²)"
                self.plot_scaling_efficiency(results, param_name, param_display, complexity, self.output_dir)

        # Statistical significance plot
        self.plot_statistical_significance(full_analysis, self.output_dir)

        print("-" * 50)
        print(f"All {self.problem_name} charts saved to: {self.output_dir}")
