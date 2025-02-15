# HPC-Driven Relational Database Operators with Advanced GPU Memory Management

This project demonstrates GPU-accelerated relational database operators (Join and Group-By) using CUDA C++. It features:
- Parallel kernels optimized for coalesced memory access
- Use of pinned memory for efficient host-to-device transfers
- Performance verification comparing CPU and GPU implementations
- Scalability tests across varying dataset sizes
- A modular design for future extensions

## Features
- **Parallel Kernels:** Implemented in `joinKernel` and `sumByKeyKernel` to ensure each thread accesses contiguous data.
- **Profiling & Optimization:** Structured to reduce warp divergence and memory contention.
- **Optimized Memory Layout:** Utilizes pinned memory (`cudaMallocHost`) for faster data transfers.
- **Performance Verification:** CPU and GPU operations are compared using C++ `<chrono>` for timing.
- **Scalability Tests:** Command-line parameters allow testing with different dataset sizes.

## Requirements
- NVIDIA CUDA Toolkit
- A compatible NVIDIA GPU

## Running the Project
 ```bash
./HPC_RelOps [numRowsA] [numRowsB] [maxKeyRange] [repeats]
```
## Example:
```bash
./HPC_RelOps 100000 100000 50000 3
```

## Project Structure
```bash
HPC_Relational_Ops/
├── README.md         # Project documentation
├── Makefile          # Build instructions
└── src/
    └── HPC_RelOps.cu # Complete C++/CUDA source code for the project
```
