# HPC-Driven Relational Database Operators with Advanced GPU Memory Management

GPU-accelerated **Hash Join** and **Group-By aggregation** in CUDA C++, optimised through iterative Nsight profiling for coalesced memory access, low warp divergence, and overlapped data transfers.

## Architecture

```
                ┌──────────────────── GPU ────────────────────┐
                │                                             │
   Stream A ────┤  H2D probe table (A)                        │
                │                                             │
                │                  ┌────────────────────┐     │
   Stream B ────┤  H2D build (B) → │  hashBuildKernel   │     │
                │                  └────────────────────┘     │
                │                          │                  │
                │            sync ─────────┘                  │
                │              │                              │
                │              ▼                              │
                │     ┌────────────────────┐                  │
                │     │  hashProbeKernel   │                  │
                │     └────────────────────┘                  │
                │              │                              │
                │              ▼                              │
                │     ┌────────────────────┐                  │
                │     │ groupByAtomicKernel│                  │
                │     └────────────────────┘                  │
                │              │                              │
                └──────────────┼──────────────────────────────┘
                               ▼
                          D2H results
```

### Key design decisions

| Technique | What it addresses |
|-----------|-------------------|
| **Hash Join** (open-addressing, `atomicCAS`) | Replaces O(n·m) nested loop with expected O(n+m); Murmur3 hash keeps probe chains short and uniform across warps |
| **SoA memory layout** | Separate `keys[]` / `vals[]` arrays → consecutive threads read consecutive addresses → coalesced 128-byte transactions |
| **Pinned memory** (`cudaMallocHost`) | Bypasses OS page-migration path, enables DMA and `cudaMemcpyAsync` |
| **Two-stream overlap** | Probe-side H2D runs concurrently with build-side H2D + hash-build kernel |
| **Atomic Group-By** | Single `atomicAdd` per thread on key-indexed accumulator — zero branch divergence |
| **Automated correctness checks** | CPU hash-map baselines serve as oracle; PASS/FAIL printed every run |

## Requirements

- NVIDIA CUDA Toolkit (≥ 11.0)
- A CUDA-capable GPU
- *(For profiling)* Nsight Systems (`nsys`) and Nsight Compute (`ncu`)

## Build & Run

```bash
make                # optimised build
make run            # run with default 100K rows
./HPC_RelOps 100000 100000 50000 5   # custom parameters
```

### Command-line parameters

```
./HPC_RelOps <numRowsA> <numRowsB> <maxKeyRange> <repeats>
```

- `numRowsA / numRowsB` — number of rows in probe / build tables
- `maxKeyRange` — integer keys drawn from `[0, maxKeyRange)`
- `repeats` — iterations for averaged timings

## Profiling

```bash
make profile        # Nsight Systems timeline trace
make ncu            # Nsight Compute kernel-level analysis
# or
chmod +x profiling/run_profiling.sh
./profiling/run_profiling.sh
```

See [`profiling/profiling_notes.md`](profiling/profiling_notes.md) for the full profiling methodology, bottleneck analysis, and the optimisations each Nsight finding motivated.

## Sample Output

```
GPU: NVIDIA GeForce RTX 3080

=== HPC Relational Ops Benchmark ===
Table A:     100000 rows
Table B:     100000 rows
Key range:   [0, 50000)
Repeats:     5

--- Join (Hash) ---
  CPU avg:  47.82 ms
  GPU avg:  14.73 ms
  Speedup:  3.25x
  Correct:  PASS

--- Group-By (Atomic) ---
  CPU avg:  5.11 ms
  GPU avg:  3.82 ms
  Speedup:  1.34x
  Correct:  PASS

--- Combined Pipeline ---
  CPU total: 52.93 ms
  GPU total: 18.55 ms
  Speedup:   2.85x

Done.
```

*(Exact numbers depend on GPU model; the combined speedup ranges from ~2–4× at 100 K rows and increases with larger datasets.)*

## Project Structure

```
HPC_Relational_Ops/
├── README.md
├── Makefile                       # build, run, profile, ncu, clean
├── src/
│   └── HPC_RelOps.cu             # All CUDA/C++ source
└── profiling/
    ├── run_profiling.sh           # Reproduce Nsight traces
    └── profiling_notes.md         # Profiling findings & optimisation log
```
