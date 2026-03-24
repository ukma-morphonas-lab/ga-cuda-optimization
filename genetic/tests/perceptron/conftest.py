import pytest
from ..logger import CUDA_GA_TestLogger
import os
from typing import Generator

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

@pytest.fixture(scope="session")
def logger(request: pytest.FixtureRequest) -> Generator[CUDA_GA_TestLogger, None, None]:
    test_file = None
    if hasattr(request, 'session') and request.session.items:
        first_item = request.session.items[0]
        test_file = str(first_item.fspath)
    
    logger = CUDA_GA_TestLogger(log_dir=LOG_DIR, test_file=test_file)
    yield logger
    logger.save_session_results()
