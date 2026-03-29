/******************************************************************************
 * HPC-Driven Relational Database Operators with Advanced GPU Memory Management
 *
 * This project implements GPU-accelerated relational database operators (Hash
 * Join and Group-By aggregation) in CUDA C++, structured around five design
 * pillars:
 *
 *   1. Parallel Kernels with Coalesced Memory Access
 *      - Structure-of-Arrays (SoA) layout keeps key[] and value[] in separate
 *        contiguous buffers so consecutive threads read consecutive addresses,
 *        maximising memory bus utilisation.
 *      - Hash Join: open-addressing hash table with Murmur-style hash to
 *        distribute keys uniformly, reducing probe-chain length variance
 *        across threads in a warp (mitigates warp divergence).
 *      - Group-By: single-pass global atomicAdd on a key-indexed accumulator,
 *        fully data-parallel with zero branch divergence.
 *
 *   2. Profiling and Optimization with Nsight Systems
 *      - See profiling/profiling_notes.md for methodology, findings, and
 *        the specific optimizations each profiling pass motivated.
 *      - Run profiling/run_profiling.sh to reproduce the Nsight traces.
 *
 *   3. Optimized Memory Layout and Transfer Pipeline
 *      - Pinned (page-locked) host memory via cudaMallocHost for DMA-capable
 *        transfers that bypass the OS page-migration path.
 *      - Two CUDA streams overlap host-to-device transfers of the probe table
 *        (stream A) with the build table transfer + hash-build kernel
 *        (stream B), hiding transfer latency behind compute.
 *
 *   4. Performance Verification
 *      - CPU baselines (hash-map join, hash-map group-by) serve as both
 *        correctness oracles and performance references.
 *      - Automated PASS/FAIL correctness checks after every run.
 *      - Speedup reported as average over configurable repeat count.
 *
 *   5. Scalability
 *      - CLI-driven dataset sizes allow sweeping from 10K to 1M+ rows.
 *
 * Compile:  make            (or: nvcc -O3 -std=c++14 src/HPC_RelOps.cu -o HPC_RelOps)
 * Run:      ./HPC_RelOps [numRowsA] [numRowsB] [maxKeyRange] [repeats]
 * Example:  ./HPC_RelOps 100000 100000 50000 5
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Utility                                                            */
/* ------------------------------------------------------------------ */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code)
                  << " (" << file << ":" << line << ")\n";
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */
constexpr int EMPTY_KEY     = -1;       // sentinel — valid keys are >= 0
constexpr int BLOCK_SIZE    = 256;
constexpr int HT_LOAD_FACTOR = 2;      // hash-table slots = load_factor * build rows

/* ------------------------------------------------------------------ */
/*  Hash function — Murmur3 32-bit finaliser                           */
/*  Distributes keys uniformly across hash-table slots, which keeps    */
/*  probe-chain lengths short and similar across threads in a warp,    */
/*  reducing warp divergence during the probe phase.                   */
/* ------------------------------------------------------------------ */
__device__ __host__ inline unsigned int murmurHash(int key, int tableSize)
{
    unsigned int h = static_cast<unsigned int>(key);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h % static_cast<unsigned int>(tableSize);
}

/* ================================================================== */
/*  CPU BASELINES (correctness oracle + performance reference)         */
/* ================================================================== */

/// CPU Hash Join — build hash map on B, probe with A.  O(n + m) expected.
void cpuHashJoin(const int* keysA, const float* valsA, int sizeA,
                 const int* keysB, const float* valsB, int sizeB,
                 std::vector<int>& outKeys, std::vector<float>& outVals)
{
    std::unordered_map<int, float> mapB;
    mapB.reserve(sizeB);
    for (int i = 0; i < sizeB; i++)
        mapB[keysB[i]] = valsB[i];          // last-write-wins on duplicates

    outKeys.resize(sizeA);
    outVals.resize(sizeA);
    for (int i = 0; i < sizeA; i++) {
        auto it = mapB.find(keysA[i]);
        if (it != mapB.end()) {
            outKeys[i] = keysA[i];
            outVals[i] = valsA[i] + it->second;
        } else {
            outKeys[i] = EMPTY_KEY;
            outVals[i] = 0.0f;
        }
    }
}

/// CPU Group-By — hash-map accumulation.  O(n) expected.
void cpuGroupBy(const int* keys, const float* vals, int numRows,
                std::unordered_map<int, float>& result)
{
    result.clear();
    result.reserve(numRows);
    for (int i = 0; i < numRows; i++)
        if (keys[i] >= 0)
            result[keys[i]] += vals[i];
}

/* ================================================================== */
/*  GPU KERNELS                                                        */
/* ================================================================== */

/* ---- Hash Join — Build Phase ------------------------------------
 * Each thread inserts one row from the build table (B) into a global
 * open-addressing hash table using atomicCAS for lock-free probing.
 * SoA layout: buildKeys[] and buildVals[] are separate contiguous
 * arrays → threads in a warp read consecutive addresses (coalesced).
 * ---------------------------------------------------------------- */
__global__
void hashBuildKernel(const int*   __restrict__ buildKeys,
                     const float* __restrict__ buildVals,
                     int buildSize,
                     int*   htKeys,
                     float* htVals,
                     int htSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= buildSize) return;

    int   key = buildKeys[idx];      // coalesced read
    float val = buildVals[idx];      // coalesced read
    unsigned int slot = murmurHash(key, htSize);

    // Linear probing with atomicCAS — no locks, no warp serialisation
    for (int i = 0; i < htSize; i++) {
        unsigned int pos = (slot + i) % htSize;
        int old = atomicCAS(&htKeys[pos], EMPTY_KEY, key);
        if (old == EMPTY_KEY || old == key) {
            htVals[pos] = val;       // last-write-wins for duplicate keys
            return;
        }
    }
    // Shouldn't reach here with HT_LOAD_FACTOR >= 2
}

/* ---- Hash Join — Probe Phase ------------------------------------
 * Each thread looks up its probe key in the hash table.
 * Reads from probeKeys/probeVals are coalesced (SoA, consecutive idx).
 * Writes to outKeys/outVals are also coalesced for the same reason.
 * Hash-table accesses are random but the Murmur hash keeps chains
 * short, so threads in a warp converge quickly (low divergence).
 * ---------------------------------------------------------------- */
__global__
void hashProbeKernel(const int*   __restrict__ probeKeys,
                     const float* __restrict__ probeVals,
                     int probeSize,
                     const int*   __restrict__ htKeys,
                     const float* __restrict__ htVals,
                     int htSize,
                     int*   outKeys,
                     float* outVals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= probeSize) return;

    int   key  = probeKeys[idx];     // coalesced
    unsigned int slot = murmurHash(key, htSize);

    for (int i = 0; i < htSize; i++) {
        unsigned int pos = (slot + i) % htSize;
        int htKey = htKeys[pos];

        if (htKey == EMPTY_KEY) {    // miss
            outKeys[idx] = EMPTY_KEY;
            outVals[idx] = 0.0f;
            return;
        }
        if (htKey == key) {          // hit
            outKeys[idx] = key;
            outVals[idx] = probeVals[idx] + htVals[pos];
            return;
        }
    }
    outKeys[idx] = EMPTY_KEY;
    outVals[idx] = 0.0f;
}

/* ---- Group-By — Atomic Aggregation ------------------------------
 * Fully data-parallel: every thread executes the same atomicAdd path
 * (no branches based on data → zero warp divergence).
 * Key-indexed output array avoids the need for sorting or segmented
 * scans.  groupFlags marks which keys are present so the host can
 * compact the result without scanning the whole key space.
 * ---------------------------------------------------------------- */
__global__
void groupByAtomicKernel(const int*   __restrict__ keys,
                         const float* __restrict__ vals,
                         int numRows,
                         float* groupSums,
                         int*   groupFlags,
                         int maxKey)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRows) return;

    int key = keys[idx];
    if (key >= 0 && key < maxKey) {
        atomicAdd(&groupSums[key], vals[idx]);
        groupFlags[key] = 1;        // mark key as present
    }
}

/* ================================================================== */
/*  GPU HOST DRIVER FUNCTIONS                                          */
/* ================================================================== */

/// GPU Hash Join with pinned memory + two-stream transfer overlap.
///
/// Stream layout:
///   streamA:  H2D(probeKeys, probeVals)
///   streamB:  H2D(buildKeys, buildVals) → hashBuildKernel
///   sync both → hashProbeKernel → D2H results
///
/// The build-side transfer and hash-table construction execute
/// concurrently with the probe-side transfer, hiding latency.
void gpuHashJoin(const int* h_keysA, const float* h_valsA, int sizeA,
                 const int* h_keysB, const float* h_valsB, int sizeB,
                 std::vector<int>& outKeys, std::vector<float>& outVals)
{
    int htSize = sizeB * HT_LOAD_FACTOR;
    outKeys.resize(sizeA);
    outVals.resize(sizeA);

    // --- Device allocations ---
    int   *d_keysA, *d_keysB, *d_outKeys, *d_htKeys;
    float *d_valsA, *d_valsB, *d_outVals, *d_htVals;

    gpuErrchk(cudaMalloc(&d_keysA,   sizeA  * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_valsA,   sizeA  * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_keysB,   sizeB  * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_valsB,   sizeB  * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_outKeys, sizeA  * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_outVals, sizeA  * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_htKeys,  htSize * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_htVals,  htSize * sizeof(float)));

    // Initialise hash table: 0xFF per byte → every int = -1 (EMPTY_KEY)
    gpuErrchk(cudaMemset(d_htKeys, 0xFF, htSize * sizeof(int)));
    gpuErrchk(cudaMemset(d_htVals, 0,    htSize * sizeof(float)));

    // --- Two-stream transfer + build overlap ---
    cudaStream_t streamA, streamB;
    gpuErrchk(cudaStreamCreate(&streamA));
    gpuErrchk(cudaStreamCreate(&streamB));

    // Stream A: probe-side transfer (A is the probe table)
    gpuErrchk(cudaMemcpyAsync(d_keysA, h_keysA, sizeA * sizeof(int),
                               cudaMemcpyHostToDevice, streamA));
    gpuErrchk(cudaMemcpyAsync(d_valsA, h_valsA, sizeA * sizeof(float),
                               cudaMemcpyHostToDevice, streamA));

    // Stream B: build-side transfer → hash-build kernel
    gpuErrchk(cudaMemcpyAsync(d_keysB, h_keysB, sizeB * sizeof(int),
                               cudaMemcpyHostToDevice, streamB));
    gpuErrchk(cudaMemcpyAsync(d_valsB, h_valsB, sizeB * sizeof(float),
                               cudaMemcpyHostToDevice, streamB));

    int gridBuild = (sizeB + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hashBuildKernel<<<gridBuild, BLOCK_SIZE, 0, streamB>>>(
        d_keysB, d_valsB, sizeB, d_htKeys, d_htVals, htSize);

    // Wait for both probe data and hash table to be ready
    gpuErrchk(cudaStreamSynchronize(streamA));
    gpuErrchk(cudaStreamSynchronize(streamB));

    // --- Probe phase (default stream) ---
    int gridProbe = (sizeA + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hashProbeKernel<<<gridProbe, BLOCK_SIZE>>>(
        d_keysA, d_valsA, sizeA,
        d_htKeys, d_htVals, htSize,
        d_outKeys, d_outVals);
    gpuErrchk(cudaDeviceSynchronize());

    // --- D2H result transfer ---
    gpuErrchk(cudaMemcpy(outKeys.data(), d_outKeys,
                          sizeA * sizeof(int),   cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(outVals.data(), d_outVals,
                          sizeA * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    gpuErrchk(cudaStreamDestroy(streamA));
    gpuErrchk(cudaStreamDestroy(streamB));
    gpuErrchk(cudaFree(d_keysA));   gpuErrchk(cudaFree(d_valsA));
    gpuErrchk(cudaFree(d_keysB));   gpuErrchk(cudaFree(d_valsB));
    gpuErrchk(cudaFree(d_outKeys)); gpuErrchk(cudaFree(d_outVals));
    gpuErrchk(cudaFree(d_htKeys));  gpuErrchk(cudaFree(d_htVals));
}

/// GPU Group-By with atomic aggregation and pinned-memory transfers.
void gpuGroupBy(const int* h_keys, const float* h_vals, int numRows,
                int maxKey, std::unordered_map<int, float>& result)
{
    int   *d_keys, *d_groupFlags;
    float *d_vals, *d_groupSums;

    gpuErrchk(cudaMalloc(&d_keys,       numRows * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_vals,       numRows * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_groupSums,  maxKey  * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_groupFlags, maxKey  * sizeof(int)));

    gpuErrchk(cudaMemset(d_groupSums,  0, maxKey * sizeof(float)));
    gpuErrchk(cudaMemset(d_groupFlags, 0, maxKey * sizeof(int)));

    gpuErrchk(cudaMemcpy(d_keys, h_keys, numRows * sizeof(int),
                          cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vals, h_vals, numRows * sizeof(float),
                          cudaMemcpyHostToDevice));

    int grid = (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    groupByAtomicKernel<<<grid, BLOCK_SIZE>>>(
        d_keys, d_vals, numRows, d_groupSums, d_groupFlags, maxKey);
    gpuErrchk(cudaDeviceSynchronize());

    // Retrieve results
    std::vector<float> sums(maxKey);
    std::vector<int>   flags(maxKey);
    gpuErrchk(cudaMemcpy(sums.data(),  d_groupSums,  maxKey * sizeof(float),
                          cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(flags.data(), d_groupFlags, maxKey * sizeof(int),
                          cudaMemcpyDeviceToHost));

    result.clear();
    for (int k = 0; k < maxKey; k++)
        if (flags[k])
            result[k] = sums[k];

    gpuErrchk(cudaFree(d_keys));
    gpuErrchk(cudaFree(d_vals));
    gpuErrchk(cudaFree(d_groupSums));
    gpuErrchk(cudaFree(d_groupFlags));
}

/* ================================================================== */
/*  CORRECTNESS VERIFICATION                                           */
/* ================================================================== */

bool verifyJoin(const std::vector<int>& cpuKeys, const std::vector<float>& cpuVals,
                const std::vector<int>& gpuKeys, const std::vector<float>& gpuVals,
                int size)
{
    int mismatches = 0;
    for (int i = 0; i < size; i++) {
        if (cpuKeys[i] != gpuKeys[i]) { mismatches++; continue; }
        if (cpuKeys[i] >= 0 && std::fabs(cpuVals[i] - gpuVals[i]) > 1e-3f)
            mismatches++;
    }
    return mismatches == 0;
}

bool verifyGroupBy(const std::unordered_map<int, float>& cpuResult,
                   const std::unordered_map<int, float>& gpuResult)
{
    if (cpuResult.size() != gpuResult.size()) return false;
    for (auto& kv : cpuResult) {
        auto it = gpuResult.find(kv.first);
        if (it == gpuResult.end()) return false;
        if (std::fabs(kv.second - it->second) > 1e-1f) return false;
    }
    return true;
}

/* ================================================================== */
/*  MAIN — Benchmarks, Verification, Scalability                       */
/* ================================================================== */
int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <numRowsA> <numRowsB> <maxKeyRange> <repeats>\n"
                  << "Example: " << argv[0] << " 100000 100000 50000 5\n";
        return EXIT_FAILURE;
    }

    int numA     = std::atoi(argv[1]);
    int numB     = std::atoi(argv[2]);
    int keyRange = std::atoi(argv[3]);
    int repeats  = std::atoi(argv[4]);

    srand(static_cast<unsigned>(time(nullptr)));

    // --- Print GPU info ---
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n\n";

    std::cout << "=== HPC Relational Ops Benchmark ===\n"
              << "Table A:     " << numA     << " rows\n"
              << "Table B:     " << numB     << " rows\n"
              << "Key range:   [0, " << keyRange << ")\n"
              << "Repeats:     " << repeats  << "\n\n";

    // --- Generate random SoA data (pinned for fast DMA transfers) ---
    int   *h_keysA, *h_keysB;
    float *h_valsA, *h_valsB;

    gpuErrchk(cudaMallocHost(&h_keysA, numA * sizeof(int)));
    gpuErrchk(cudaMallocHost(&h_valsA, numA * sizeof(float)));
    gpuErrchk(cudaMallocHost(&h_keysB, numB * sizeof(int)));
    gpuErrchk(cudaMallocHost(&h_valsB, numB * sizeof(float)));

    for (int i = 0; i < numA; i++) {
        h_keysA[i] = rand() % keyRange;
        h_valsA[i] = static_cast<float>(rand() % 1000) / 10.0f;
    }
    for (int i = 0; i < numB; i++) {
        h_keysB[i] = rand() % keyRange;
        h_valsB[i] = static_cast<float>(rand() % 1000) / 10.0f;
    }

    // --- Warm-up GPU (first CUDA call has driver-init overhead) ---
    {
        void* tmp;
        gpuErrchk(cudaMalloc(&tmp, 1));
        gpuErrchk(cudaFree(tmp));
    }

    // --- Benchmark loop ---
    double totalCpuJoin = 0, totalGpuJoin = 0;
    double totalCpuGroup = 0, totalGpuGroup = 0;
    bool joinCorrect = true, groupCorrect = true;

    for (int r = 0; r < repeats; r++) {

        // ---- CPU Join ----
        std::vector<int>   cpuJoinKeys;
        std::vector<float> cpuJoinVals;
        auto t0 = std::chrono::high_resolution_clock::now();
        cpuHashJoin(h_keysA, h_valsA, numA,
                    h_keysB, h_valsB, numB,
                    cpuJoinKeys, cpuJoinVals);
        auto t1 = std::chrono::high_resolution_clock::now();
        totalCpuJoin += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- GPU Join ----
        std::vector<int>   gpuJoinKeys;
        std::vector<float> gpuJoinVals;
        t0 = std::chrono::high_resolution_clock::now();
        gpuHashJoin(h_keysA, h_valsA, numA,
                    h_keysB, h_valsB, numB,
                    gpuJoinKeys, gpuJoinVals);
        t1 = std::chrono::high_resolution_clock::now();
        totalGpuJoin += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- Verify Join ----
        if (!verifyJoin(cpuJoinKeys, cpuJoinVals, gpuJoinKeys, gpuJoinVals, numA))
            joinCorrect = false;

        // ---- CPU Group-By (on join output) ----
        std::unordered_map<int, float> cpuGroupResult;
        t0 = std::chrono::high_resolution_clock::now();
        cpuGroupBy(cpuJoinKeys.data(), cpuJoinVals.data(), numA, cpuGroupResult);
        t1 = std::chrono::high_resolution_clock::now();
        totalCpuGroup += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- GPU Group-By (on join output) ----
        std::unordered_map<int, float> gpuGroupResult;
        t0 = std::chrono::high_resolution_clock::now();
        gpuGroupBy(gpuJoinKeys.data(), gpuJoinVals.data(), numA,
                   keyRange, gpuGroupResult);
        t1 = std::chrono::high_resolution_clock::now();
        totalGpuGroup += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- Verify Group-By ----
        if (!verifyGroupBy(cpuGroupResult, gpuGroupResult))
            groupCorrect = false;
    }

    // --- Report ---
    double avgCJ = totalCpuJoin  / repeats;
    double avgGJ = totalGpuJoin  / repeats;
    double avgCG = totalCpuGroup / repeats;
    double avgGG = totalGpuGroup / repeats;

    double joinSpeedup  = (avgGJ > 0) ? avgCJ / avgGJ : 0;
    double groupSpeedup = (avgGG > 0) ? avgCG / avgGG : 0;
    double totalCpu = avgCJ + avgCG;
    double totalGpu = avgGJ + avgGG;
    double totalSpeedup = (totalGpu > 0) ? totalCpu / totalGpu : 0;

    std::cout << "--- Join (Hash) ---\n"
              << "  CPU avg:  " << avgCJ << " ms\n"
              << "  GPU avg:  " << avgGJ << " ms\n"
              << "  Speedup:  " << joinSpeedup << "x\n"
              << "  Correct:  " << (joinCorrect ? "PASS" : "FAIL") << "\n\n";

    std::cout << "--- Group-By (Atomic) ---\n"
              << "  CPU avg:  " << avgCG << " ms\n"
              << "  GPU avg:  " << avgGG << " ms\n"
              << "  Speedup:  " << groupSpeedup << "x\n"
              << "  Correct:  " << (groupCorrect ? "PASS" : "FAIL") << "\n\n";

    std::cout << "--- Combined Pipeline ---\n"
              << "  CPU total: " << totalCpu << " ms\n"
              << "  GPU total: " << totalGpu << " ms\n"
              << "  Speedup:   " << totalSpeedup << "x\n\n";

    // --- Free pinned memory ---
    gpuErrchk(cudaFreeHost(h_keysA));
    gpuErrchk(cudaFreeHost(h_valsA));
    gpuErrchk(cudaFreeHost(h_keysB));
    gpuErrchk(cudaFreeHost(h_valsB));

    std::cout << "Done.\n";
    return EXIT_SUCCESS;
}
