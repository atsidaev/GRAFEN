# Validation tests. Default: CPU. GPU=1 links real CUDA gFieldSolver (no MPI).
#
#   make -C src/tests test                      # CPU, demag on
#   make -C src/tests test ARGS='--no-demag'    # skip demag solve
#   make -C src/tests GPU=1 test                # CUDA
#
# Main binary (cluster): add -noDemag or -demag 0 to disable demag CG.
#
# Optional overrides for the cluster:
#   make GPU=1 CUDA_PATH=/usr/local/cuda-12 CUDA_ARCH=sm_70 test

CXX ?= g++
NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= compute_35
ARGS ?=

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
INCLUDES = -I..
NVCC_FLAGS ?= --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -arch $(CUDA_ARCH)

GPU ?= 0

.PHONY: all test clean

all: grafen_tests

ifeq ($(GPU),1)

NVCC_BIN := $(if $(wildcard $(CUDA_PATH)/bin/nvcc),$(CUDA_PATH)/bin/nvcc,$(NVCC))

grafen_tests: tests_main.o calcField.o
	$(NVCC_BIN) -o $@ $^ $(CXXFLAGS) -lrt

tests_main.o: tests_main.cpp field_cpu.h mesh_cpu.h analytic.h check.h
	$(CXX) -c -o $@ tests_main.cpp $(CXXFLAGS) $(INCLUDES) -DUSE_CUDA -I$(CUDA_PATH)/include

calcField.o: ../calcField.cu ../calcField.h ../mobj.h ../cuVar.h
	$(NVCC_BIN) -c -o $@ ../calcField.cu $(CXXFLAGS) $(NVCC_FLAGS) $(INCLUDES)

else

grafen_tests: tests_main.cpp field_cpu.h mesh_cpu.h analytic.h check.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ tests_main.cpp

endif

test: grafen_tests
	./grafen_tests $(ARGS)

clean:
	rm -f grafen_tests tests_main.o calcField.o
