import inspect
import json
import logging
import platform
import psutil

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
from cuda.profiling.gpu_state_snapshot import GPUStateSnapshot


DEFAULT_LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

class CUDA_GA_TestLogger:
    def __init__(self, log_dir: str = DEFAULT_LOGS_DIR, test_file: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.results_dir = self.log_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.test_file = self._resolve_test_file(test_file)
        self.test_file_stem = self.test_file.stem if self.test_file else "unknown_test"

        self._setup_logging()

        self.session_start = datetime.now().isoformat()
        self.environment_info = self._get_environment_info()
        self.test_results: List[Dict[str, Any]] = []
        self.current_test: Optional[Dict[str, Any]] = None

        self._log_session_info()

    def _resolve_test_file(self, test_file: Optional[str]) -> Path:
        if test_file:
            return Path(test_file).resolve()

        for frame in inspect.stack():
            candidate = Path(frame.filename)
            if candidate.suffix == ".py" and candidate.name.startswith("test_"):
                return candidate.resolve()
        
        pytest_current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
        if pytest_current_test:
            # "path/to/test_file.py::test_name (call)"
            test_path = pytest_current_test.split("::")[0]
            if test_path:
                return Path(test_path).resolve()

        return Path("unknown_test.py")

    def _setup_logging(self):
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.log_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.run_dir / f"{self.test_file_stem}_test_run_{self.timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        # console_handler = logging.StreamHandler()  # Removed to prevent stdout spam

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)  # Removed to prevent stdout spam

    def _get_environment_info(self) -> Dict[str, Any]:
        env_info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": 1,
            "memory_total": 0,
            "memory_available": 0,
            "memory_used_percent": 0.0,
            "gpu_count": 0,
            "gpu_name": "N/A",
            "gpu_memory_total": 0,
            "gpu_memory_available": 0,
            "gpu_memory_used_percent": 0.0
        }

        mem_info = psutil.virtual_memory()
        cpu_count = psutil.cpu_count() or 1
        env_info.update({
            "cpu_count": cpu_count,
            "memory_total": mem_info.total // (1024**3),  # GB
            "memory_available": mem_info.available // (1024**3),  # GB
            "memory_used_percent": mem_info.percent,
        })

        gpu_snapshot = GPUStateSnapshot()
        env_info.update(gpu_snapshot.provide_snapshot())

        return env_info

    def _log_session_info(self):
        self.logger.info("=" * 80)
        self.logger.info("TEST SESSION ENVIRONMENT INFORMATION")
        self.logger.info("=" * 80)

        env = self.environment_info
        self.logger.info(f"Python: {env['python_version']}")
        self.logger.info(f"Platform: {env['platform']}")
        self.logger.info(f"CPU cores: {env['cpu_count']}")
        self.logger.info(f"Memory: {env['memory_total']}GB total, {env['memory_available']}GB available")

        if env['gpu_count'] > 0:
            self.logger.info(f"GPU: {env['gpu_name']} ({env['gpu_memory_total']}GB)")
            self.logger.info(f"GPU Memory: {env['gpu_memory_available']}GB available, {env['gpu_memory_used_percent']:.1f}% used")

        self.logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
    

    def start_test(self, test_name: str) -> None:
        if self.current_test:
            self.logger.warning("Previous test not finished, ending it automatically")
            self.end_test("interrupted")

        self.current_test = {
            "test_name": test_name,
            "test_start_time": datetime.now().isoformat(),
            "test_parameters": None,
            "test_output": "",
            "test_error": "",
            "test_traceback": "",
            "test_result": "running"
        }

        self.logger.info(f"Starting test: {test_name}")

    def log_param(self, key: str, value: Any) -> None:
        if not self.current_test:
            self.logger.warning("No active test, ignoring parameter log")
            return

        if self.current_test["test_parameters"] is None:
            self.current_test["test_parameters"] = {}

        self.current_test["test_parameters"][key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_output(self, message: str) -> None:
        if not self.current_test:
            self.logger.warning("No active test, ignoring output log")
            return

        if self.current_test["test_output"]:
            self.current_test["test_output"] += "\n"
        self.current_test["test_output"] += str(message)

    def end_test(self, result: str, error: str = "", traceback: str = "", extra_info: Dict[str, Any] = {}) -> None:
        if not self.current_test:
            self.logger.warning("No active test to end")
            return

        self.current_test.update({
            "test_end_time": datetime.now().isoformat(),
            "test_result": result,
            "test_error": error,
            "test_traceback": traceback,
            "test_extra_info": extra_info
        })

        # duration
        start = datetime.fromisoformat(self.current_test["test_start_time"])
        end = datetime.fromisoformat(self.current_test["test_end_time"])
        self.current_test["test_duration"] = (end - start).total_seconds()

        if self.current_test["test_parameters"]:
            self.test_results.append(self.current_test.copy())

        self.logger.info(f"Test {self.current_test['test_name']} completed: {result} ({self.current_test['test_duration']:.2f}s)")
        self.current_test = None
        

    def save_session_results(self) -> None:
        session_end = datetime.now().isoformat()

        session_start_dt = datetime.fromisoformat(self.session_start)
        session_end_dt = datetime.fromisoformat(session_end)
        session_duration = (session_end_dt - session_start_dt).total_seconds()

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["test_result"] == "passed"])
        failed_tests = total_tests - passed_tests

        session_data = {
            "session_start_time": self.session_start,
            "session_end_time": session_end,
            "session_duration": session_duration,
            "environment_info": self.environment_info,
            "test_file": str(self.test_file),
            "total_tests_run": total_tests,
            "tests_passed": passed_tests,
            "tests_failed": failed_tests,
            "test_results": self.test_results
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = self.results_dir / f"{self.test_file_stem}_session_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        self.logger.info("=" * 80)
        self.logger.info("SESSION COMPLETE")
        self.logger.info(f"Results saved to: {filename}")
        self.logger.info(f"Tests run: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")
        self.logger.info(f"Session duration: {session_duration:.2f}s")
        self.logger.info("=" * 80)