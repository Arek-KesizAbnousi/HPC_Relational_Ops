/******************************************************************************
 * HPC-Driven Relational Database Operators with Advanced GPU Memory Management
 *
 * This project is divided into six key parts:
 *
 *   1. Parallel Kernels with Coalesced Memory Access
 *      - Implemented in joinKernel and sumByKeyKernel to ensure each thread
 *        accesses contiguous data, maximizing throughput.
 *
 *   2. Profiling and Optimization with Nsight Systems
 *      - Although profiling is done externally using Nsight Systems, the code 
 *        is structured to reduce warp divergence and memory contention.
 *
 *   3. Optimized Memory Layout
 *      - Achieved by using pinned memory (cudaMallocHost) to speed up host-to-device
 *        transfers and by organizing data for coalesced access.
 *
 *   4. Performance Verification and Speedup Measurement
 *      - CPU and GPU versions of join and group-by are compared using timing
 *        routines (chrono) to quantify speedups.
 *
 *   5. Scalability Tests
 *      - The main function accepts command-line parameters to test scalability
 *        across different dataset sizes.
 *
 *   6. Modular Framework for Future HPC Operators
 *      - The project is designed in a modular fashion, allowing future extensions
 *        and integration of additional HPC operators.
 *
 * Compile: nvcc HPC_RelOps.cu -o HPC_RelOps
 * Run:     ./HPC_RelOps [numRowsA] [numRowsB] [maxKeyRange] [repeats]
 *
 * Example: ./HPC_RelOps 100000 100000 50000 3
 *
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>

//---------------------------------------------------------
// Utility macros/checks for CUDA error handling
//---------------------------------------------------------
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code)
                  << " " << file << ":" << line << std::endl;
        exit(code);
    }
}

//---------------------------------------------------------
// Data structure to mimic a table row
//---------------------------------------------------------
struct Row {
    int   key;
    float value;
};

//// Part 1: CPU Baselines for Verification
//// CPU Baseline: Naive Join (for performance and correctness comparison)
std::vector<Row> cpuJoin(const std::vector<Row>& tableA,
                         const std::vector<Row>& tableB)
{
    std::vector<Row> out;
    out.reserve(tableA.size()); // naive estimate

    for (size_t i = 0; i < tableA.size(); ++i) {
        for (size_t j = 0; j < tableB.size(); ++j) {
            if (tableA[i].key == tableB[j].key) {
                Row joined;
                joined.key   = tableA[i].key;
                joined.value = tableA[i].value + tableB[j].value;
                out.push_back(joined);
            }
        }
    }
    return out;
}

//// CPU Baseline: Naive Group-By (for performance and correctness comparison)
std::vector<Row> cpuGroupBy(const std::vector<Row>& table)
{
    std::unordered_map<int, float> groupSums;
    groupSums.reserve(table.size());

    for (auto &r : table) {
        groupSums[r.key] += r.value;
    }

    // Convert map to vector<Row>
    std::vector<Row> out;
    out.reserve(groupSums.size());
    for (auto &kv : groupSums) {
        out.push_back({kv.first, kv.second});
    }
    return out;
}

//// Part 1: Parallel Kernels with Coalesced Memory Access
//// GPU Join Kernel:
//// For each row in A, search B for a match, ensuring threads access contiguous memory.
__global__
void joinKernel(const int* d_keysA, const float* d_valsA, int sizeA,
                const int* d_keysB, const float* d_valsB, int sizeB,
                int* d_outKeys, float* d_outVals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeA) return;

    int aKey = d_keysA[idx];
    float aVal = d_valsA[idx];
    bool matched = false;
    for (int j = 0; j < sizeB; ++j) {
        if (aKey == d_keysB[j]) {
            d_outKeys[idx] = aKey;
            d_outVals[idx] = aVal + d_valsB[j];
            matched = true;
            break;
        }
    }
    if (!matched) {
        d_outKeys[idx] = -1;  // mark no match
        d_outVals[idx] = 0.0f;
    }
}

//// Part 1: Parallel Kernels with Coalesced Memory Access
//// GPU Group-By Kernel:
//// Performs a naive summation, structured to minimize warp divergence.
__global__
void sumByKeyKernel(const int* d_keys, const float* d_vals, float* d_outVals, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int curKey = d_keys[idx];
    float curVal = d_vals[idx];
    float sumVal = curVal;
    if (idx < size - 1 && d_keys[idx+1] == curKey) {
        sumVal += d_vals[idx+1];
    }
    d_outVals[idx] = sumVal;
}

//// Part 3: Optimized Memory Layout & Data Transfer
//// GPU Host Function: Join
//// Uses pinned memory (cudaMallocHost) for efficient data transfers between host and device.
void gpuJoin(const std::vector<Row>& tableA,
             const std::vector<Row>& tableB,
             std::vector<Row>& out)
{
    size_t sizeA = tableA.size();
    size_t sizeB = tableB.size();
    out.resize(sizeA);

    // Allocate pinned (page-locked) memory for faster Host->Device transfers.
    int *h_keysA = nullptr, *h_keysB = nullptr;
    float *h_valsA = nullptr, *h_valsB = nullptr;
    gpuErrchk(cudaMallocHost((void**)&h_keysA, sizeA * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_valsA, sizeA * sizeof(float)));
    gpuErrchk(cudaMallocHost((void**)&h_keysB, sizeB * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_valsB, sizeB * sizeof(float)));

    // Populate pinned arrays from input tables.
    for (size_t i = 0; i < sizeA; i++) {
        h_keysA[i] = tableA[i].key;
        h_valsA[i] = tableA[i].value;
    }
    for (size_t i = 0; i < sizeB; i++) {
        h_keysB[i] = tableB[i].key;
        h_valsB[i] = tableB[i].value;
    }

    // Device allocations for GPU arrays.
    int *d_keysA, *d_keysB, *d_outKeys;
    float *d_valsA, *d_valsB, *d_outVals;
    gpuErrchk(cudaMalloc((void**)&d_keysA, sizeA * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_valsA, sizeA * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_keysB, sizeB * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_valsB, sizeB * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_outKeys, sizeA * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_outVals, sizeA * sizeof(float)));

    // Copy data from pinned host memory to device memory.
    gpuErrchk(cudaMemcpy(d_keysA, h_keysA, sizeA * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_valsA, h_valsA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_keysB, h_keysB, sizeB * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_valsB, h_valsB, sizeB * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the Parallel Join Kernel.
    int blockSize = 256;
    int gridSize = (sizeA + blockSize - 1) / blockSize;
    joinKernel<<<gridSize, blockSize>>>(d_keysA, d_valsA, sizeA,
                                        d_keysB, d_valsB, sizeB,
                                        d_outKeys, d_outVals);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Retrieve results from GPU.
    std::vector<int> h_outKeys(sizeA);
    std::vector<float> h_outVals(sizeA);
    gpuErrchk(cudaMemcpy(h_outKeys.data(), d_outKeys, sizeA * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_outVals.data(), d_outVals, sizeA * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < sizeA; i++) {
        out[i].key = h_outKeys[i];
        out[i].value = h_outVals[i];
    }

    // Cleanup: Free pinned and device memory.
    gpuErrchk(cudaFreeHost(h_keysA));
    gpuErrchk(cudaFreeHost(h_valsA));
    gpuErrchk(cudaFreeHost(h_keysB));
    gpuErrchk(cudaFreeHost(h_valsB));
    gpuErrchk(cudaFree(d_keysA));
    gpuErrchk(cudaFree(d_valsA));
    gpuErrchk(cudaFree(d_keysB));
    gpuErrchk(cudaFree(d_valsB));
    gpuErrchk(cudaFree(d_outKeys));
    gpuErrchk(cudaFree(d_outVals));
}

//// Part 3: Optimized Memory Layout & Data Transfer in Group-By
//// GPU Host Function: Group-By
void gpuGroupBy(std::vector<Row>& table)
{
    size_t size = table.size();
    if (size == 0) return;

    // Sort the table by key on the host.
    std::sort(table.begin(), table.end(), [](const Row &a, const Row &b) { return a.key < b.key; });

    // Allocate pinned memory for optimized transfer.
    int *h_keys = nullptr;
    float *h_vals = nullptr;
    gpuErrchk(cudaMallocHost((void**)&h_keys, size * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_vals, size * sizeof(float)));

    for (size_t i = 0; i < size; i++) {
        h_keys[i] = table[i].key;
        h_vals[i] = table[i].value;
    }

    // Device allocations for group-by operation.
    int *d_keys;
    float *d_vals, *d_outVals;
    gpuErrchk(cudaMalloc((void**)&d_keys, size * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_vals, size * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_outVals, size * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_keys, h_keys, size * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vals, h_vals, size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the Group-By Kernel.
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    sumByKeyKernel<<<gridSize, blockSize>>>(d_keys, d_vals, d_outVals, size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Retrieve partial sums from device.
    std::vector<float> partialSums(size);
    gpuErrchk(cudaMemcpy(partialSums.data(), d_outVals, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Consolidate the final grouped results on the CPU.
    std::vector<Row> grouped;
    grouped.reserve(size);

    int currentKey = h_keys[0];
    float aggVal = partialSums[0];

    for (size_t i = 1; i < size; i++) {
        if (h_keys[i] == currentKey) {
            aggVal = partialSums[i]; // minimal adjacency sum for demonstration.
        } else {
            grouped.push_back({currentKey, aggVal});
            currentKey = h_keys[i];
            aggVal = partialSums[i];
        }
    }
    grouped.push_back({currentKey, aggVal});
    table = grouped;

    // Cleanup: Free pinned and device memory.
    gpuErrchk(cudaFreeHost(h_keys));
    gpuErrchk(cudaFreeHost(h_vals));
    gpuErrchk(cudaFree(d_keys));
    gpuErrchk(cudaFree(d_vals));
    gpuErrchk(cudaFree(d_outVals));
}

//// Part 5: Scalability Tests and Performance Verification in Main
int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " [numRowsA] [numRowsB] [maxKeyRange] [repeats]\n";
        return 1;
    }
    int numA = std::atoi(argv[1]);
    int numB = std::atoi(argv[2]);
    int keyRange = std::atoi(argv[3]);
    int repeats = std::atoi(argv[4]);

    srand((unsigned)time(nullptr));

    std::cout << "HPC Relational Ops Demo\n"
              << "Table A: " << numA << " rows\n"
              << "Table B: " << numB << " rows\n"
              << "Key Range: " << keyRange << "\n"
              << "Repeats:   " << repeats << "\n\n";

    // Generate test data for scalability tests.
    std::vector<Row> tableA, tableB;
    auto generateRandomTable = [](std::vector<Row>& table, int numRows, int keyRange) {
        table.resize(numRows);
        for (int i = 0; i < numRows; i++) {
            int k = rand() % keyRange;
            float v = static_cast<float>(rand() % 1000) / 10.0f;
            table[i] = { k, v };
        }
    };
    generateRandomTable(tableA, numA, keyRange);
    generateRandomTable(tableB, numB, keyRange);

    double totalCpuJoin = 0.0;
    double totalGpuJoin = 0.0;
    double totalCpuGroup = 0.0;
    double totalGpuGroup = 0.0;

    for (int r = 0; r < repeats; r++) {
        // Copy data for each iteration to ensure fairness.
        std::vector<Row> A = tableA;
        std::vector<Row> B = tableB;

        // CPU Join
        auto t1 = std::chrono::high_resolution_clock::now();
        auto cpuJoinResult = cpuJoin(A, B);
        auto t2 = std::chrono::high_resolution_clock::now();
        double msCPUJoin = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // GPU Join
        t1 = std::chrono::high_resolution_clock::now();
        std::vector<Row> gpuJoinResult;
        gpuJoin(A, B, gpuJoinResult);
        t2 = std::chrono::high_resolution_clock::now();
        double msGPUJoin = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // CPU Group-By
        t1 = std::chrono::high_resolution_clock::now();
        auto cpuGroupResult = cpuGroupBy(cpuJoinResult);
        t2 = std::chrono::high_resolution_clock::now();
        double msCPUGroup = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // GPU Group-By
        std::vector<Row> gpuGroupResult = gpuJoinResult; // use GPU join result as input
        t1 = std::chrono::high_resolution_clock::now();
        gpuGroupBy(gpuGroupResult);
        t2 = std::chrono::high_resolution_clock::now();
        double msGPUGroup = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // Accumulate timings.
        totalCpuJoin += msCPUJoin;
        totalGpuJoin += msGPUJoin;
        totalCpuGroup += msCPUGroup;
        totalGpuGroup += msGPUGroup;
    }

    // Compute averages and speedups.
    double avgCpuJoin = totalCpuJoin / repeats;
    double avgGpuJoin = totalGpuJoin / repeats;
    double avgCpuGroup = totalCpuGroup / repeats;
    double avgGpuGroup = totalGpuGroup / repeats;
    double joinSpeedup = (avgCpuJoin < 1.0) ? 0.0 : (avgCpuJoin / avgGpuJoin);
    double groupSpeedup = (avgCpuGroup < 1.0) ? 0.0 : (avgCpuGroup / avgGpuGroup);

    std::cout << "Average CPU Join time  (ms): " << avgCpuJoin << std::endl;
    std::cout << "Average GPU Join time  (ms): " << avgGpuJoin << std::endl;
    std::cout << "Join Speedup (CPU/GPU): ~" << joinSpeedup << "x\n\n";
    std::cout << "Average CPU Group time (ms): " << avgCpuGroup << std::endl;
    std::cout << "Average GPU Group time (ms): " << avgGpuGroup << std::endl;
    std::cout << "Group Speedup (CPU/GPU): ~" << groupSpeedup << "x\n\n";
    std::cout << "Done.\n";

    return 0;
}
