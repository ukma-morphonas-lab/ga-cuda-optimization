import sys
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any
from genetic.tests.visualization_base import GAVisualizer


# TODO: fix "no module named genetic" error more cleanly
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent.parent 
sys.path.insert(0, str(project_root))



class TSPVisualizer(GAVisualizer):
    def __init__(self, logs_dir: Path, output_dir: Path):
        super().__init__("TSP", logs_dir, output_dir)

    def get_scaling_parameters(self) -> List[Tuple[str, List[Any], str]]:
        return [
            ("graph_size", [20, 50, 100, 200], "Number of Cities"),
            ("population_size", [50, 100, 200, 500], "Population Size")
        ]

    def load_scaling_results(self, param_name: str, param_values: List[Any]) -> List[Dict[str, Any]]:
        results = []

        for param_val in param_values:
            pattern = "statistical_analysis_tsp_*runs_"
            matching_files = list(self.logs_dir.glob(f"{pattern}*.json"))

            for f in matching_files:
                try:
                    with open(f) as fp:
                        data = json.load(fp)

                        if param_name == "graph_size":
                            if (data['config']['graph_size'] == param_val and
                                data['config']['population_size'] == 100 and
                                data['config']['max_generations'] == 100):
                                results.append(data)
                                break
                        elif param_name == "population_size":
                            if (data['config']['population_size'] == param_val and
                                data['config']['graph_size'] == 50 and
                                data['config']['max_generations'] == 100):
                                results.append(data)
                                break
                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    continue

        results.sort(key=lambda x: x['config'][param_name])
        return results

    def load_full_analysis(self) -> Dict[str, Any]:
        pattern = "statistical_analysis_tsp_30runs_"
        files = list(self.logs_dir.glob(f"{pattern}*.json"))
        if files:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            with open(files[0]) as f:
                return json.load(f)
        return {}


def main():
    script_dir = Path(__file__).parent
    logs_base_dir = script_dir.parent / 'logs'

    # Find the latest run_* directory
    run_dirs = list(logs_base_dir.glob('run_*'))
    if not run_dirs:
        print(f"No run_* directories found in {logs_base_dir}")
        return

    # Sort by modification time (newest first) and pick the latest
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    logs_dir = run_dirs[0]
    print(f"Using latest run directory: {logs_dir}")

    output_dir = script_dir
    visualizer = TSPVisualizer(logs_dir, output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
