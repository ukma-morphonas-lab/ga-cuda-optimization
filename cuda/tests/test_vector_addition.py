import pytest
import numpy as np
import cupy as cp
from numba import cuda, njit


N = 10**7  # vector size
cpu_array_a = np.random.random(N).astype(np.float32)
cpu_array_b = np.random.random(N).astype(np.float32)

# configure numba data
numba_array_a = cuda.to_device(cpu_array_a)
numba_array_b = cuda.to_device(cpu_array_b)
numba_array_out = cuda.device_array_like(numba_array_a)

threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block


# configure cupy data
cupy_array_a = cp.asarray(cpu_array_a)
cupy_array_b = cp.asarray(cpu_array_b)


# CPU (NumPy)
def numpy_cpu(a, b):
    return a + b

# CPU (Numba njit parallel)
@njit
def numba_cpu(a, b):
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = a[i] + b[i]
    return out

# Numba CUDA kernel
@cuda.jit
def vector_add_kernel(a, b, out):
    thread_id = cuda.grid(1) # type: ignore - temporary suppression
    if thread_id < a.size:
        out[thread_id] = a[thread_id] + b[thread_id]

def numba_gpu():
    vector_add_kernel[blocks_per_grid, threads_per_block](numba_array_a, numba_array_b, numba_array_out) # type: ignore - temporary suppression
    cuda.synchronize()

# CuPy GPU
def cupy_gpu():
    _ = cupy_array_a + cupy_array_b
    cp.cuda.Stream.null.synchronize()

@pytest.mark.benchmark
def test_cpu_numpy(benchmark):
    benchmark(numpy_cpu, cpu_array_a, cpu_array_b)

@pytest.mark.benchmark
def test_cpu_numba(benchmark):
    benchmark(numba_cpu, cpu_array_a, cpu_array_b)

@pytest.mark.benchmark
def test_gpu_numba(benchmark):
    benchmark(numba_gpu)

@pytest.mark.benchmark
def test_gpu_cupy(benchmark):
    benchmark(cupy_gpu)
