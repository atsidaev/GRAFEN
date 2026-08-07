# VTU → Surfer GRD via CUDA gFieldSolver (no demagnetization).
#
#   make -C src/tools
#   make -C src/tools CUDA_PATH=/usr/local/cuda CUDA_ARCH=sm_86
#
#   ./vtu_to_grd model.vtu out.grd --x=-20,20,10 --y=-20,20,10 -z 0 -H 2 2 2 -k 1

TOOLS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC := $(abspath $(TOOLS_DIR)/..)

CXX ?= g++
NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= compute_35

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
INCLUDES = -I$(SRC) -I$(CUDA_PATH)/include
NVCC_FLAGS ?= --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -arch $(CUDA_ARCH)

NVCC_BIN := $(if $(wildcard $(CUDA_PATH)/bin/nvcc),$(CUDA_PATH)/bin/nvcc,$(NVCC))

.PHONY: all clean

all: vtu_to_grd

vtu_to_grd: vtu_to_grd.o calcField.o Grid.o
	$(NVCC_BIN) -o $@ $^ $(CXXFLAGS) -lrt

vtu_to_grd.o: $(TOOLS_DIR)/vtu_to_grd.cpp $(SRC)/tests/model_vtu.h $(SRC)/calcField.h $(SRC)/mobj.h $(SRC)/Grid/Grid.h
	$(CXX) -c -o $@ $(TOOLS_DIR)/vtu_to_grd.cpp $(CXXFLAGS) $(INCLUDES) -DUSE_CUDA

calcField.o: $(SRC)/calcField.cu $(SRC)/calcField.h $(SRC)/mobj.h $(SRC)/cuVar.h
	$(NVCC_BIN) -c -o $@ $(SRC)/calcField.cu $(CXXFLAGS) $(NVCC_FLAGS) -I$(SRC)

Grid.o: $(SRC)/Grid/Grid.cpp $(SRC)/Grid/Grid.h
	$(CXX) -c -o $@ $(SRC)/Grid/Grid.cpp $(CXXFLAGS) -I$(SRC)/Grid -I$(SRC)

clean:
	rm -f vtu_to_grd vtu_to_grd.o calcField.o Grid.o
