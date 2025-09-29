# Module 05 – CUDA Memory Assignment

This assignment demonstrates the use of **all CUDA memory types** with
≥64 threads and includes timing comparisons, CLI argument usage,
and reproducible build/run scripts.

---

## Project Structure
- **src/memory.cu** – Main CUDA program:
  - Host memory allocation & initialization
  - Global memory operations
  - Constant memory usage (`__constant__` variables)
  - Shared memory tiling and reduction
  - Register memory demo kernel
  - Command-line argument parsing (`-n`, `-b`, `-k`, `-r`)
- **Makefile** – Build script (compiles to `build/memory`)
- **run.sh** – Runs program with multiple block sizes and N values
- **build_log.txt** – Captured build output (`ptxas info` register usage)
- **run_log.txt** – Captured run output with timing results

---

## Build & Run Instructions

### Build
```bash
make
```

Captures register counts (e.g., ptxas info : Used 12 registers).

Run
```
./run.sh | tee run_log.txt
```

Or run manually with custom arguments:
```
./build/memory -n 1048576 -b 128 -k shared -r 3
```

Where:
- -n = number of elements
- -b = threads per block
- -k = kernel variant (global, constant, shared, all)
- -r = number of timing repetitions

### Key Observations
- **Constant memory** consistently shows slight performance improvements compared to global memory.
- **Shared memory** adds a small overhead from synchronization but enables per-block reduction operations, which would scale better for large reductions.