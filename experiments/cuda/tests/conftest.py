import pytest
import logging
import cupy as cp
import platform
import psutil
import subprocess
import sys
from datetime import datetime
import os


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"logs/test_run_{timestamp}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Keep console output too
    ]
)

logger = logging.getLogger(__name__)

def get_nvcc_version():
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line:
                    return line.strip()
        return "nvcc not found"
    except Exception as e:
        return f"Error getting nvcc version: {e}"

def get_gpu_info():
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        gpu_info = []
        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            gpu_info.append({
                'device_id': i,
                'name': props['name'].decode('utf-8'),
                'total_memory': props['totalGlobalMem'] // (1024**3),  # GB
                'multiprocessors': props['multiProcessorCount'],
                'compute_capability': f"{props['major']}.{props['minor']}"
            })
        return gpu_info
    except Exception as e:
        return f"Error getting GPU info: {e}"

def get_memory_info():
    try:
        mem = psutil.virtual_memory()
        return {
            'total': mem.total // (1024**3),  # GB
            'available': mem.available // (1024**3),  # GB
            'used_percent': mem.percent
        }
    except Exception as e:
        return f"Error getting memory info: {e}"

@pytest.fixture(scope="session", autouse=True)
def log_environment_info():
    logger.info("=" * 80)
    logger.info("TEST SESSION ENVIRONMENT INFORMATION")
    logger.info("=" * 80)

    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")

    mem_info = get_memory_info()
    if isinstance(mem_info, dict):
        logger.info(f"System Memory - Total: {mem_info['total']}GB, Available: {mem_info['available']}GB, Used: {mem_info['used_percent']}%")
    else:
        logger.info(f"System Memory: {mem_info}")

    # CUDA/CuPy
    try:
        logger.info(f"CuPy version: {cp.__version__}")
        cuda_version = cp.cuda.get_local_runtime_version()
        logger.info(f"CUDA Runtime version: {cuda_version}")
        driver_version = cp.cuda.runtime.driverGetVersion()
        logger.info(f"CUDA Driver version: {driver_version}")
    except Exception as e:
        logger.info(f"CUDA/CuPy info error: {e}")

    # NVCC version
    nvcc_version = get_nvcc_version()
    logger.info(f"NVCC version: {nvcc_version}")

    # GPU 
    gpu_info = get_gpu_info()
    if isinstance(gpu_info, list):
        for gpu in gpu_info:
            logger.info(f"GPU {gpu['device_id']}: {gpu['name']}")
            logger.info(f"  Memory: {gpu['total_memory']}GB")
            logger.info(f"  Multiprocessors: {gpu['multiprocessors']}")
            logger.info(f"  Compute Capability: {gpu['compute_capability']}")
    else:
        logger.info(f"GPU info: {gpu_info}")

    logger.info(f"Test session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    yield

    logger.info("=" * 80)
    logger.info(f"Test session completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
