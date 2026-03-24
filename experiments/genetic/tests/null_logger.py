import logging
from typing import Any, Dict


class NullLogger:
    """
    Null object logger that satisfies CUDA_GA_TestLogger interface
    without performing any I/O operations.

    Use this for batch statistical runs where logging overhead is undesirable.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"NullLogger_{id(self)}")
        self._logger.addHandler(logging.NullHandler())
        self._logger.setLevel(logging.CRITICAL + 1)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def start_test(self, test_name: str) -> None:
        pass

    def log_param(self, key: str, value: Any) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_output(self, message: str) -> None:
        pass

    def end_test(self, result: str, error: str = "",
                 traceback: str = "", extra_info: Dict[str, Any] | None = None) -> None:
        pass

    def save_session_results(self) -> None:
        pass


def create_null_logger() -> NullLogger:
    return NullLogger()
