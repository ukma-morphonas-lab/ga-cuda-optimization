import numpy as np
import numba
from numba import cuda
import math
import pytest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#! This benchmarks version is first demo WIP and is committed for tracking progress.
#! It is not yet ready for use.

#? About warmup: https://forums.developer.nvidia.com/t/why-warm-up/48565/2 
#? Tile size: https://stackoverflow.com/questions/64466437/estimating-the-optimal-tiling-size-for-gpu-matrix-computations 

@cuda.jit
def matmul_kernel_basic(A, B, C):
    #* Simple matrix multiplication kernel without shared memory

    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # type: ignore
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore
    
    # matrix dimensions: A(rows_A × inner_dim) @ B(inner_dim × cols_B) = C(rows_A × cols_B)
    rows_A, inner_dim = A.shape
    _, cols_B = B.shape
    
    if row >= rows_A or col >= cols_B:    
        return
    
    product = 0.0
    for k in range(inner_dim):
        product += A[row, k] * B[k, col]
    C[row, col] = product


@cuda.jit
def matmul_kernel_shared(A, B, C):
    #* Matrix multiplication kernel using shared memory
    TILE_SIZE = 16 # TODO: try 6 or 8
    
    # shared memory arrays - one tile of A and B per block
    shared_arr_A = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32) # type: ignore
    shared_arr_B = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32) # type: ignore
    
    # thread position
    thread_x = cuda.threadIdx.x
    thread_y = cuda.threadIdx.y
    
    # output position
    row = cuda.blockIdx.y * TILE_SIZE + thread_y  # type: ignore
    col = cuda.blockIdx.x * TILE_SIZE + thread_x  # type: ignore
    
    product = 0.0
    # matrix dimensions: A(rows_A × inner_dim) @ B(inner_dim × cols_B) = C(rows_A × cols_B)
    rows_A, inner_dim = A.shape
    _, cols_B = B.shape
    num_tiles = (inner_dim + TILE_SIZE - 1) // TILE_SIZE
    
    for tile in range(num_tiles):
        # load one tile of A into shared memory
        a_col = tile * TILE_SIZE + thread_x  # type: ignore
        if row < rows_A and a_col < inner_dim:
            shared_arr_A[thread_y, thread_x] = A[row, a_col]  # type: ignore
        else:
            shared_arr_A[thread_y, thread_x] = 0.0  # type: ignore
        
        # load one tile of B into shared memory
        b_row = tile * TILE_SIZE + thread_y  # type: ignore
        if b_row < inner_dim and col < cols_B:
            shared_arr_B[thread_y, thread_x] = B[b_row, col]  # type: ignore
        else:
            shared_arr_B[thread_y, thread_x] = 0.0  # type: ignore
        
        # wait for all threads to load data into shared memory
        cuda.syncthreads() # type: ignore
        
        # compute partial dot product using shared memory
        for k in range(TILE_SIZE):
            product += shared_arr_A[thread_y, k] * shared_arr_B[k, thread_x]  # type: ignore
        
        # wait for all threads to compute partial dot product
        cuda.syncthreads() # type: ignore
    
    # write result
    if row < rows_A and col < cols_B:
        C[row, col] = product


def matmul_gpu(A: np.ndarray, B: np.ndarray, block_size: tuple[int, int] = (16,16), verbose: bool = False) -> np.ndarray:
    # TODO: why do we need to make the arrays contiguous? (from numba exception)
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    
    rows_A, inner_dim_A = A.shape
    inner_dim_B, cols_B = B.shape
    assert inner_dim_A == inner_dim_B, f"Matrix dimensions don't match: A is {rows_A}x{inner_dim_A}, B is {inner_dim_B}x{cols_B}"
    
    # allocate output matrix
    C = np.zeros((rows_A, cols_B), dtype=np.float32)
    
    # transfer data to GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    
    # configure thread blocks and grid
    threads_per_block = block_size
    
    # calculate optimal number of blocks
    blocks_per_grid_x = math.ceil(cols_B / threads_per_block[0])
    blocks_per_grid_y = math.ceil(rows_A / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # launch kernel
    matmul_kernel_basic[blocks_per_grid, threads_per_block](d_A, d_B, d_C) # type: ignore
    
    # copy result to CPU
    result = d_C.copy_to_host()
    
    return result


def matmul_gpu_shared(A: np.ndarray, B: np.ndarray, block_size: tuple[int, int] = (16,16), verbose: bool = False) -> np.ndarray:
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    
    rows_A, inner_dim_A = A.shape
    inner_dim_B, cols_B = B.shape
    
    assert inner_dim_A == inner_dim_B, f"Matrix dimensions don't match: A is {rows_A}x{inner_dim_A}, B is {inner_dim_B}x{cols_B}"
    assert block_size == (16,16)
    
    # allocate output matrix
    C = np.zeros((rows_A, cols_B), dtype=np.float32)
    
    # transfer to GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    
    # configure thread blocks and grid
    threads_per_block = block_size
    
    # calculate optimal number of blocks
    blocks_per_grid_x = math.ceil(cols_B / threads_per_block[0])
    blocks_per_grid_y = math.ceil(rows_A / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    total_blocks = blocks_per_grid[0] * blocks_per_grid[1]
    logger.info(f"Shared memory kernel: {total_blocks} blocks")
    
    # launch kernel
    matmul_kernel_shared[blocks_per_grid, threads_per_block](d_A, d_B, d_C) # type: ignore
    
    result = d_C.copy_to_host()
    return result

class TestMatrixMultiplication:
    @pytest.fixture(scope="class")
    def warmup(self) -> None:
        A_demo = np.random.randn(16,16).astype(np.float32)
        B_demo = np.random.randn(16,16).astype(np.float32)
        _ = matmul_gpu(A_demo, B_demo)
        _ = matmul_gpu_shared(A_demo, B_demo)
        cuda.synchronize()
    
    @pytest.mark.parametrize("rows_A,inner_dim,cols_B", [
        (16,16,16),      # low occupancy
        (64, 48, 64),      # small
        (128, 128, 128),   # medium
        (256, 128, 256),   # medium-large
        (512, 512, 512),   # large
    ])
    def test_basic_kernel(self, warmup, rows_A, inner_dim, cols_B):
        A = np.random.randn(rows_A, inner_dim).astype(np.float32)
        B = np.random.randn(inner_dim, cols_B).astype(np.float32)
        
        C_gpu = matmul_gpu(A, B, block_size=(16,16))
        C_cpu = np.matmul(A, B)
        
        assert np.allclose(C_gpu, C_cpu, rtol=1e-4, atol=1e-5), \
            f"Results don't match for {rows_A}x{inner_dim} @ {inner_dim}x{cols_B}"
    
    @pytest.mark.parametrize("rows_A,inner_dim,cols_B", [
        (16,16,16),
        (64, 48, 64),
        (128, 128, 128),
        (256, 128, 256),
        (512, 512, 512),
    ])
    def test_shared_kernel(self, warmup, rows_A, inner_dim, cols_B):
        A = np.random.randn(rows_A, inner_dim).astype(np.float32)
        B = np.random.randn(inner_dim, cols_B).astype(np.float32)
        
        C_gpu = matmul_gpu_shared(A, B, block_size=(16,16))
        C_cpu = np.matmul(A, B)
        
        assert np.allclose(C_gpu, C_cpu, rtol=1e-4, atol=1e-5), \
            f"Results don't match for {rows_A}x{inner_dim} @ {inner_dim}x{cols_B}"
    
    @pytest.mark.parametrize("block_size", [
        (8, 8),
        (16, 16),
        (32, 32),
    ])
    def test_block_sizes(self, warmup, block_size):
        A = np.random.randn(256, 128).astype(np.float32)
        B = np.random.randn(128, 256).astype(np.float32)
        
        C_gpu = matmul_gpu(A, B, block_size=block_size)
        C_cpu = np.matmul(A, B)
        
        assert np.allclose(C_gpu, C_cpu, rtol=1e-4, atol=1e-5)
    
    def test_non_square_matrices(self, warmup):
        test_cases = [
            (100, 50, 75),   # all different dimensions
            (200, 100, 200), # rows_A == cols_B
            (100, 200, 100), # rows_A == cols_B, inner_dim different
        ]
        
        for rows_A, inner_dim, cols_B in test_cases:
            A = np.random.randn(rows_A, inner_dim).astype(np.float32)
            B = np.random.randn(inner_dim, cols_B).astype(np.float32)
            
            C_gpu = matmul_gpu(A, B)
            C_cpu = np.matmul(A, B)
            
            assert np.allclose(C_gpu, C_cpu, rtol=1e-4, atol=1e-5), \
                f"Failed for dimensions {rows_A}x{inner_dim} @ {inner_dim}x{cols_B}"


@pytest.fixture(scope="module")
def benchmark_matrices():
    np.random.seed(42)
    sizes = {
        '256': (256, 256, 256),
        '512': (512, 512, 512),
        '1024': (1024, 1024, 1024),
    }
    
    matrices = {}
    for name, (rows_A, inner_dim, cols_B) in sizes.items():
        A = np.ascontiguousarray(np.random.randn(rows_A, inner_dim).astype(np.float32))
        B = np.ascontiguousarray(np.random.randn(inner_dim, cols_B).astype(np.float32))
        matrices[name] = (A, B)
    
    # warmup
    A_demo, B_demo = matrices['256']
    _ = matmul_gpu(A_demo[:32, :32], B_demo[:32, :32])
    _ = matmul_gpu_shared(A_demo[:32, :32], B_demo[:32, :32])
    cuda.synchronize()
    
    return matrices


class TestPerformanceBenchmarks:
    @staticmethod
    def _run_matmul_gpu(A, B, block_size=(16,16)):
        result = matmul_gpu(A, B, block_size=block_size)
        cuda.synchronize()
        return result
    
    @staticmethod
    def _run_matmul_gpu_shared(A, B, block_size=(16,16)):
        result = matmul_gpu_shared(A, B, block_size=block_size)
        cuda.synchronize()
        return result
    
    @pytest.mark.benchmark
    def test_benchmark_basic_small(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['256']
        result = benchmark(lambda: self._run_matmul_gpu(A, B))
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_shared_small(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['256']
        result = benchmark(lambda: self._run_matmul_gpu_shared(A, B))
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_numpy_small(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['256']
        result = benchmark(np.matmul, A, B)
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_basic_medium(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['512']
        result = benchmark(lambda: self._run_matmul_gpu(A, B))
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_shared_medium(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['512']
        result = benchmark(lambda: self._run_matmul_gpu_shared(A, B))
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_numpy_medium(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['512']
        result = benchmark(np.matmul, A, B)
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_basic_large(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['1024']
        result = benchmark(lambda: self._run_matmul_gpu(A, B))
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_shared_large(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['1024']
        result = benchmark(lambda: self._run_matmul_gpu_shared(A, B))
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)
    
    @pytest.mark.benchmark
    def test_benchmark_numpy_large(self, benchmark, benchmark_matrices):
        A, B = benchmark_matrices['1024']
        result = benchmark(np.matmul, A, B)
        C_cpu = np.matmul(A, B)
        assert np.allclose(result, C_cpu, atol=1e-4)