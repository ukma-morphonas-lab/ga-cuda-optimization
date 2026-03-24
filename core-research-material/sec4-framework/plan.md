# Section 4: Framework Implementation

## 4.1 — Library Design and Architecture (`core-lib`)

- **Motivation:** extracting reusable, modular GPU-GA infrastructure from the incremental research codebase (`genetic/tests/`) into a standalone library
- **Package structure:** `kernels/`, `device/`, `statistics/`, `artifacts/`
- **Key design patterns:**
  - **Strategy pattern** for fitness evaluation — `FitnessStrategy` lets users plug in problem-specific fitness kernels while the library handles memory, grid config, and orchestration
  - **Facade pattern** — `GAMemoryManager`, `RNGManager`, `FitnessEvaluator` hide CUDA boilerplate (pinned allocation, RNG state init, grid config) behind simple interfaces
  - **Paired memory abstraction** — `PinnedDevicePair` encapsulates host-pinned + device buffer pairs, enforcing the pinned-memory optimization by construction
  - **Composable kernel modules** — `SelectionKernels`, `CrossoverKernels` (single-point, two-point, uniform), `MutationKernels` (gaussian, swap, inversion, bit-flip), `PopulationKernels` — each independently usable
- **Automatic occupancy-based grid configuration** via `device/` module (CC 8.6 heuristics from Section 3.1.2)
- **Statistical analysis module** mirrors the paired experimental methodology from Section 3 — `run_experiment()` / `run_full_analysis()` with identical seed control, paired tests, bootstrap CI
- **Flexibility:** fitness kernel left user-defined; library provides infrastructure, not problem-specific logic

## 4.2 — Reproduced Experiment Results and Comparison with Section 3

- Re-run both benchmarks (XOR perceptron, TSP) using `core-lib` API instead of the original incremental test code
- Compare reproduced results against Section 3 baselines:
  - **XOR:** wall-time speedup (~1.51x at pop=100), scaling curve (0.69x at pop=50 up to 14.11x at pop=1000), transfer overhead (~29.2%)
  - **TSP:** speedup (~1.15x at 50 cities/pop=200), scaling behavior, transfer overhead reduction (38.6% → 17.7%)
- Verify that the refactored library preserves identical statistical properties (p-values, effect sizes, fitness distributions)
- Discuss any performance differences — overhead from abstraction layers vs. raw inline kernels
- Demonstrate that the library's modular design does not introduce measurable regression in throughput or speedup ratios
