import pytest
from ..logger import CUDA_GA_TestLogger
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

@pytest.fixture(scope="session", autouse=True)
def logger() -> CUDA_GA_TestLogger:
    return CUDA_GA_TestLogger(log_dir=LOG_DIR)
