ROOT_DIR ?= $(PWD)/../../llama.cpp
BUILD_DIR ?= build
OPT ?= -O3
NOGPU ?=

layer-gpu-bench: layer-compute-bench.o layer-cpu-compute.o layer-gpu-compute.o
	$(CXX) $^ -L$(ROOT_DIR)/$(BUILD_DIR)/bin -L/usr/local/cuda/lib64 -lllama -lggml-cpu -lggml-cuda -lggml -lmtmd -lggml-base -lgomp -lcudart -lcublas -lcuda -o $@

layer-cpu-bench: layer-compute-bench.o layer-cpu-compute.o
	$(CXX) $^ -L$(ROOT_DIR)/cpu_build/bin -lllama -lggml-cpu -lggml -lmtmd -lggml-base -lgomp -o $@

layer-compute-bench.o: layer-compute-bench.cpp layer-compute-bench.h
	$(CXX) $(NOGPU) -I$(ROOT_DIR)/src -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/common -I$(ROOT_DIR)/vendor -I$(ROOT_DIR)/include -I$(ROOT_DIR)/ggml/include -I$(ROOT_DIR)/ggml -I$(ROOT_DIR)/ggml/src/ggml-cpu -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -c $< -o $@ $(OPT)

layer-cpu-compute.o: layer-compute.c layer-compute.h
	$(CC) -I$(ROOT_DIR)/ggml -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/ggml/src/ggml-cpu -I$(ROOT_DIR)/ggml/include -g -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -march=native -fopenmp -std=gnu11 -c $< -o $@ $(OPT)

layer-gpu-compute.o: layer-compute.cu layer-compute.cuh
	nvcc -forward-unknown-to-host-compiler -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=1 -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cuda_EXPORTS -I$(ROOT_DIR)/ggml/include -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/ggml/src/ggml-cuda -isystem=/usr/local/cuda/include --generate-code=arch=compute_70,code=[compute_70] --generate-code=arch=compute_75,code=[compute_75] --generate-code=arch=compute_80,code=[compute_80] --generate-code=arch=compute_86,code=[sm_86] --generate-code=arch=compute_89,code=[sm_89] -Xcompiler=-fPIC -use_fast_math -extended-lambda -Xcompiler -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -Wno-pedantic -std=c++17 -x cu -c $< -o $@ $(OPT)

clean:
	rm -rf *.o layer-gpu-bench layer-cpu-bench
