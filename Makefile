# Makefile — HPC-Driven Relational Database Operators
# Supports: build, debug build, Nsight Systems profiling, clean

NVCC     = nvcc
CFLAGS   = -O3 -std=c++14
DBGFLAGS = -G -g -lineinfo -std=c++14
SRC      = src/HPC_RelOps.cu
TARGET   = HPC_RelOps

# Default dataset parameters for quick benchmarks
ROWS_A   ?= 100000
ROWS_B   ?= 100000
KEY_RANGE?= 50000
REPEATS  ?= 5

# ---- Build targets ----
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET)

debug: $(SRC)
	$(NVCC) $(DBGFLAGS) $(SRC) -o $(TARGET)_debug

# ---- Run ----
run: $(TARGET)
	./$(TARGET) $(ROWS_A) $(ROWS_B) $(KEY_RANGE) $(REPEATS)

# ---- Nsight Systems profiling ----
profile: $(TARGET)
	nsys profile \
	  --trace=cuda,nvtx,osrt \
	  --output=profiling/nsys_report \
	  --force-overwrite=true \
	  ./$(TARGET) $(ROWS_A) $(ROWS_B) $(KEY_RANGE) $(REPEATS)
	@echo ""
	@echo "Trace saved to profiling/nsys_report.nsys-rep"
	@echo "Open with: nsys-ui profiling/nsys_report.nsys-rep"

# ---- Nsight Compute kernel-level profiling ----
ncu: $(TARGET)
	ncu --set full \
	  --target-processes all \
	  -o profiling/ncu_report \
	  ./$(TARGET) $(ROWS_A) $(ROWS_B) $(KEY_RANGE) 1
	@echo ""
	@echo "Report saved to profiling/ncu_report.ncu-rep"

# ---- Clean ----
clean:
	rm -f $(TARGET) $(TARGET)_debug
	rm -f profiling/*.nsys-rep profiling/*.sqlite profiling/*.ncu-rep

.PHONY: all debug run profile ncu clean
