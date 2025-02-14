# Makefile for HPC-Driven Relational Database Operators

# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -std=c++11

# Target executable name
TARGET = HPC_RelOps

# Source file location
SRC = src/HPC_RelOps.cu

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)
