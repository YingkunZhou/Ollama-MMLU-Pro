ROOT_DIR ?= $(PWD)/../../llama.cpp
BUILD_DIR ?= build
OPT ?= -O3

layer-compute-bench: layer-compute-bench.o layer-compute.o layer-compute.cu.o
	$(CXX) $^ -L$(ROOT_DIR)/$(BUILD_DIR)/bin -lllama -lggml-cpu -lggml-cuda -lggml -lmtmd -lggml-base -lgomp -o $@

layer-compute.o: layer-compute.c layer-compute.h
	$(CC) -I$(ROOT_DIR)/ggml -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/ggml/src/ggml-cpu -I$(ROOT_DIR)/ggml/include -g -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -march=native -fopenmp -std=gnu11 -c $< -o $@ $(OPT)

layer-compute-bench.o: layer-compute-bench.cpp layer-compute-bench.h
	$(CXX) -I$(ROOT_DIR)/src -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/common -I$(ROOT_DIR)/vendor -I$(ROOT_DIR)/include -I$(ROOT_DIR)/ggml/include -I$(ROOT_DIR)/ggml -I$(ROOT_DIR)/ggml/src/ggml-cpu -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -c $< -o $@ $(OPT)

layer-compute.cu.o: layer-compute.cu
	nvcc -forward-unknown-to-host-compiler -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=1 -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cuda_EXPORTS -I$(ROOT_DIR)/ggml/include -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/ggml/src/ggml-cuda -isystem=/usr/local/cuda-12.6/include $(OPT) -DNDEBUG --generate-code=arch=compute_70,code=[compute_70] --generate-code=arch=compute_75,code=[compute_75] --generate-code=arch=compute_80,code=[compute_80] --generate-code=arch=compute_86,code=[sm_86] --generate-code=arch=compute_89,code=[sm_89] -Xcompiler=-fPIC -use_fast_math -extended-lambda -Xcompiler -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -Wno-pedantic -std=c++17 -x cu -c $< -o $@

clean:
	rm *.o layer-compute-bench
