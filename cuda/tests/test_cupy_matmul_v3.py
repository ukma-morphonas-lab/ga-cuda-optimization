import cupy as cp
import time
from typing import TypeAlias
import pytest
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Matrix: TypeAlias = cp.ndarray


class TestPerformanceBenchmarks:

    @pytest.fixture(scope="module", params=[2048, 4096, 8192, 10000])
    def random_matrix_pair(self, request):
        size = request.param

        np_dtype = np.float32 if size >= 4096 else np.float64
        cp_dtype = cp.float32 if size >= 4096 else cp.float64

        numpy_matrix = np.random.rand(size, size).astype(np_dtype)
        cupy_matrix = cp.random.rand(size, size, dtype=cp_dtype)

        return {
            'size': size,
            'numpy': numpy_matrix,
            'cupy': cupy_matrix
        }
    
    
    def _matmul_gpu(self, mat_a: Matrix, mat_b: Matrix) -> Matrix:
        result = cp.matmul(mat_a, mat_b)
        cp.cuda.Stream.null.synchronize()
        return result
    
    # https://www.kdnuggets.com/leveraging-the-power-of-gpus-with-cupy-in-python 
    # Extra demo benchmark for reference
    def test_demo_numpy(self):
        s = time.time()
        x_cpu = np.ones((1000, 100, 1000))
        np.sqrt(np.sum(x_cpu**2, axis=-1))
        e = time.time()
        np_time = e - s
        logger.info(f"Demo ones matrix multiplication with NumPy: Time consumed: {np_time} seconds")
        
    def test_demo_cupy(self):
        s = time.time()
        x_gpu = cp.ones((1000, 100, 1000))
        cp.sqrt(cp.sum(x_gpu**2, axis=-1))
        e = time.time()
        cp_time = e - s
        logger.info(f"Demo ones matrix multiplication with CuPy: Time consumed: {cp_time} seconds")
        

    @pytest.mark.parametrize("library", ["numpy", "cupy"])
    def test_benchmark_matrix_multiplication(self, benchmark, random_matrix_pair, library):
        size = random_matrix_pair['size']
        mat_a = random_matrix_pair[library]
        mat_b = random_matrix_pair[library]

        start_time = time.time()

        if library == "cupy":
            self._matmul_gpu(mat_a, mat_b)
        else:
            np.matmul(mat_a, mat_b)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{library.capitalize()} random {size}x{size} matrix multiplication: Time taken: {duration:.6f} seconds")
        