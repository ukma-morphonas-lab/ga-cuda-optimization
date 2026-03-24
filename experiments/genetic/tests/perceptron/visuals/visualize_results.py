from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
from genetic.tests.visualization_base import GAVisualizer

class PerceptronVisualizer(GAVisualizer):
    def __init__(self, logs_dir: Path, output_dir: Path):
        super().__init__("Perceptron", logs_dir, output_dir)

    def get_scaling_parameters(self) -> List[Tuple[str, List[Any], str]]:
        return [
            ("population_size", [50, 100, 500, 1000], "Population Size")
        ]

    def load_scaling_results(self, param_name: str, param_values: List[Any]) -> List[Dict[str, Any]]:
        results = []

        for pop_size in param_values:
            pattern = f"statistical_analysis_10runs_"
            matching_files = list(self.logs_dir.glob(f"{pattern}*.json"))

            for f in matching_files:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        if (data['config']['population_size'] == pop_size and
                            data['config']['max_generations'] == 200):
                            results.append(data)
                            break
                except:
                    continue

        # Sort by population size
        results.sort(key=lambda x: x['config']['population_size'])
        return results

    def load_full_analysis(self) -> Dict[str, Any]:
        pattern = "statistical_analysis_30runs_"
        files = list(self.logs_dir.glob(f"{pattern}*.json"))
        if files:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            with open(files[0]) as f:
                return json.load(f)
        return {}

def main():
    script_dir = Path(__file__).parent
    logs_dir = script_dir.parent / 'logs'
    output_dir = script_dir
    visualizer = PerceptronVisualizer(logs_dir, output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
