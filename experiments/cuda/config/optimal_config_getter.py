from dataclasses import dataclass
import cupy as cp
import logging

logger = logging.getLogger(__name__)


ThreadsPerSM = int
ThreadsPerBlock = int
BlocksPerGrid = int
Occupancy = float
PopulationSize = int
TotalGPUThreads = int

# Max threads per SM varies by compute capability
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

# 8.6
MAX_THREADS_PER_SM_8_6 = 1536
MAX_WARPS_PER_SM_8_6 = 64
MAX_BLOCKS_PER_SM_8_6 = 32
MAX_GRID_SIZE_PER_DEVICE_8_6 = 128
MAX_THREADS_PER_BLOCK_8_6 = 1024
WARP_SIZE_8_6 = 32

@dataclass
class GPUProperties:
    name: str
    major: int
    minor: int
    multiProcessorCount: int
    maxThreadsPerBlock: int
    warpSize: int
    totalGlobalMem: int
    sharedMemPerBlock: int
    regsPerBlock: int
    maxThreadsDim: tuple[int, int, int]
    maxGridSize: tuple[int, int, int]
    clockRate: int
    memoryClockRate: int
    memoryBusWidth: int
    l2CacheSize: int
    concurrentKernels: bool
    ECCEnabled: bool
    integrated: bool
    canMapHostMemory: bool
    
    @property
    def compute_capability(self) -> tuple[int, int]:
        return (self.major, self.minor)
    
    @property
    def max_threads_per_sm(self) -> ThreadsPerSM:
        match self.compute_capability:
            case (8, 6):
                return MAX_THREADS_PER_SM_8_6
            case _:
                logger.warning(f"Unknown compute capability: {self.compute_capability}")
                return 2048
    
    @property
    def total_gpu_threads(self) -> TotalGPUThreads:
        return self.multiProcessorCount * self.max_threads_per_sm

    
def get_gpu_properties() -> GPUProperties:
    device_id = cp.cuda.Device().id
    raw_props = cp.cuda.runtime.getDeviceProperties(device_id)
    
    return GPUProperties(
        name=raw_props['name'].decode('utf-8'),
        major=raw_props['major'],
        minor=raw_props['minor'],
        multiProcessorCount=raw_props['multiProcessorCount'],
        maxThreadsPerBlock=raw_props['maxThreadsPerBlock'],
        warpSize=raw_props['warpSize'],
        totalGlobalMem=raw_props['totalGlobalMem'],
        sharedMemPerBlock=raw_props['sharedMemPerBlock'],
        regsPerBlock=raw_props['regsPerBlock'],
        maxThreadsDim=tuple(raw_props['maxThreadsDim']), 
        maxGridSize=tuple(raw_props['maxGridSize']), 
        clockRate=raw_props['clockRate'],
        memoryClockRate=raw_props['memoryClockRate'],
        memoryBusWidth=raw_props['memoryBusWidth'],
        l2CacheSize=raw_props['l2CacheSize'],
        concurrentKernels=bool(raw_props['concurrentKernels']), 
        ECCEnabled=bool(raw_props['ECCEnabled']),      
        integrated=bool(raw_props['integrated']),              
        canMapHostMemory=bool(raw_props['canMapHostMemory']),     
    )


def calculate_occupancy(total_allocated_threads: int, total_gpu_threads: int) -> Occupancy:
    return (total_allocated_threads / total_gpu_threads) * 100


def get_threads_per_block(population_size: PopulationSize, warp_size: int, max_threads_per_block: int) -> ThreadsPerBlock:
    MIN_POPULATION_SIZE = 256
    MAX_POPULATION_SIZE = 10000
    
    if population_size < MIN_POPULATION_SIZE:
        return ((population_size + warp_size - 1) // warp_size) * warp_size
    elif population_size >= MAX_POPULATION_SIZE:
        return min(512, max_threads_per_block)
    else:
        return 256
    
def get_blocks_per_grid(population_size: PopulationSize, threads_per_block: ThreadsPerBlock, max_threads_per_sm: int) -> BlocksPerGrid:
    return min(max_threads_per_sm, (population_size + threads_per_block - 1) // threads_per_block)

def estimate_occupancy(occupancy: Occupancy) -> None:
    match occupancy:
        case _ if occupancy < 25:
            logger.warning("Very low GPU occupancy")
        case _ if occupancy < 50:
            logger.warning("Low GPU occupancy - GPU may be underutilized")
        case _ if occupancy < 80:
            logger.info("Moderate GPU occupancy - acceptable performance")
        case _ if occupancy < 100:
            logger.info("Good GPU occupancy - efficient GPU utilization")
        case _ if occupancy > 100:
            logger.warning("Overloaded GPU - GPU may be overloaded")


def calculate_optimal_threads_config(
    population_size: PopulationSize, 
    verbose: bool = True, 
    props: GPUProperties | None = None
) -> tuple[ThreadsPerBlock, BlocksPerGrid]:
    if props is None:
        props = get_gpu_properties()
    
    max_threads_per_block: int = props.maxThreadsPerBlock
    warp_size: int = props.warpSize
    total_gpu_threads: int = props.total_gpu_threads
    max_threads_per_sm: int = props.max_threads_per_sm
    
    threads_per_block: ThreadsPerBlock = get_threads_per_block(population_size, warp_size, max_threads_per_block)
    blocks_per_grid: BlocksPerGrid = get_blocks_per_grid(population_size, threads_per_block, max_threads_per_sm)
    total_allocated_threads: int = blocks_per_grid * threads_per_block
    occupancy: Occupancy = calculate_occupancy(total_allocated_threads, total_gpu_threads)
    
    if verbose:
        logger.info("RECOMMENDED THREADS CONFIGURATION")
        logger.info("="*70)
        logger.info(f"GPU: {props.name}")
        logger.info(f"Compute Capability: {props.major}.{props.minor}")
        logger.info(f"Population size: {population_size:,}")
        logger.info(f"Threads per block: {threads_per_block}")
        logger.info(f"Blocks per grid: {blocks_per_grid}")
        logger.info(f"Total threads launched: {total_allocated_threads:,}")
        logger.info(f"GPU capacity: {total_gpu_threads:,}")
        logger.info(f"GPU occupancy: {occupancy:.1f}%")
        logger.info(f"Active warps per block: {threads_per_block // warp_size}")
        
        estimate_occupancy(occupancy)
        logger.info("="*70)
    
    return threads_per_block, blocks_per_grid


def calculate_optimal_generation_size(
    num_cities: int, 
    gpu_capacity: TotalGPUThreads | None = None, 
    verbose: bool = True, 
    gpu_properties: GPUProperties | None = None
) -> PopulationSize:
    if gpu_capacity is None:
        if gpu_properties is None:
            gpu_properties = get_gpu_properties()
        total_gpu_threads: int = gpu_properties.total_gpu_threads
    
    # TARGET: 50-100% GPU occupancy
    # multiple of 256 for optimal performance
    optimal_population: PopulationSize = (total_gpu_threads // 256) * 256
    
    # heuristic: population = 10-50x problem size
    heuristic_min: PopulationSize = num_cities * 10
    heuristic_max: PopulationSize = num_cities * 50
    
    if optimal_population < heuristic_min:
        recommended = ((heuristic_min + 255) // 256) * 256
    elif optimal_population > heuristic_max:
        recommended = ((heuristic_max + 255) // 256) * 256
    else:
        recommended = optimal_population
    
    if verbose:
        logger.info("RECOMMENDED GENERATION SIZE")
        logger.info(f"Number of cities: {num_cities}")
        logger.info(f"GPU capacity: {total_gpu_threads:,} threads")
        logger.info(f"GPU-based recommendation: {optimal_population:,}")
        logger.info(f"GA heuristic range: {heuristic_min:,} - {heuristic_max:,}")
        logger.info(f"RECOMMENDED: {recommended:,}")
        logger.info("  (Multiple of 256 for optimal GPU performance)")
        logger.info("="*70)
    
    return recommended


def analyze_current_config(
    population_size: PopulationSize, 
    threads_per_block: ThreadsPerBlock = 256, 
    max_threads_per_sm: int = MAX_THREADS_PER_SM_8_6,
    props: GPUProperties | None = None
) -> Occupancy:
    if props is None:
        props = get_gpu_properties()
    
    total_gpu_threads: int = props.total_gpu_threads
    blocks_per_grid: BlocksPerGrid = get_blocks_per_grid(population_size, threads_per_block, max_threads_per_sm)
    total_allocated_threads: int = blocks_per_grid * threads_per_block
    occupancy: Occupancy = calculate_occupancy(total_allocated_threads, total_gpu_threads)
     
    logger.info(f"GPU: {props.name}")
    logger.info(f"Population: {population_size:,}")
    logger.info(f"Threads per block: {threads_per_block}")
    logger.info(f"Blocks per grid: {blocks_per_grid}")
    logger.info(f"Total threads: {total_allocated_threads:,}")
    logger.info(f"GPU capacity: {total_gpu_threads:,}")
    logger.info(f"Occupancy: {occupancy:.1f}%")
    logger.info("="*70)
    return occupancy


def print_all_gpu_info(props: GPUProperties | None = None) -> None:
    if props is None:
        props = get_gpu_properties()
    
    logger.info("COMPLETE GPU DEVICE PROPERTIES")
    logger.info("="*70)
    
    info_items = [
        ("Device Name", props.name),
        ("Compute Capability", f"{props.major}.{props.minor}"),
        ("Total Global Memory", f"{props.totalGlobalMem / (1024**3):.2f} GB"),
        ("Shared Memory Per Block", f"{props.sharedMemPerBlock / 1024:.2f} KB"),
        ("Registers Per Block", props.regsPerBlock),
        ("Warp Size", props.warpSize),
        ("Max Threads Per Block", props.maxThreadsPerBlock),
        ("Max Threads Per SM", props.max_threads_per_sm),
        ("Total GPU Threads", f"{props.total_gpu_threads:,}"),
        ("Max Block Dimensions", f"({props.maxThreadsDim[0]}, {props.maxThreadsDim[1]}, {props.maxThreadsDim[2]})"),
        ("Max Grid Dimensions", f"({props.maxGridSize[0]}, {props.maxGridSize[1]}, {props.maxGridSize[2]})"),
        ("Clock Rate", f"{props.clockRate / 1000:.2f} MHz"),
        ("Memory Clock Rate", f"{props.memoryClockRate / 1000:.2f} MHz"),
        ("Memory Bus Width", f"{props.memoryBusWidth} bits"),
        ("L2 Cache Size", f"{props.l2CacheSize / 1024:.2f} KB"),
        ("Multiprocessor Count", props.multiProcessorCount),
        ("Concurrent Kernels", "Yes" if props.concurrentKernels else "No"),
        ("ECC Enabled", "Yes" if props.ECCEnabled else "No"),
        ("Integrated", "Yes" if props.integrated else "No"),
        ("Can Map Host Memory", "Yes" if props.canMapHostMemory else "No"),
    ]
    
    for name, value in info_items:
        logger.info(f"{name:.<35} {value}")
    
    logger.info("="*70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    logger.info("CUDA GPU Configuration Optimizer for TSP GA (using CuPy)")
    logger.info("="*70)
    
    try:
        print_all_gpu_info()
    except Exception as e:
        logger.error(f"ERROR: Could not get GPU properties: {e}")
        exit(1)
    
    # Test cases from pytest parameters (currently TSP cities, population)
    test_cases = [
        # (10, 100),
        # (100, 1000),
        # (500, 5000),
        (500, 10000),
        (500, 20000),
        (500, 23000),
        (1000, 23000),
    ]
    
    for num_cities, population_size in test_cases:
        logger.info(f"TEST CASE: {num_cities} cities, population {population_size}")
        logger.info("="*70)
        
        occupancy: Occupancy = analyze_current_config(population_size)
        
        if occupancy < 50:
            logger.info("Low GPU occupancy - calculating optimal population size...")
            threads, blocks = calculate_optimal_threads_config(population_size)
            calculate_optimal_generation_size(num_cities)