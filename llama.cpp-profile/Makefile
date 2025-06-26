ROOT_DIR ?= $(PWD)/../../llama.cpp
BUILD_DIR ?= build
OPT ?= -O3

layer-compute-bench: layer-compute-bench.o layer-compute.o
	$(CXX) $^ -L$(ROOT_DIR)/$(BUILD_DIR)/bin -lllama -lggml-cpu -lggml -lmtmd -lggml-base -lgomp -o $@

layer-compute.o: layer-compute.c layer-compute.h
	$(CC) -I$(ROOT_DIR)/ggml -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/ggml/src/ggml-cpu -I$(ROOT_DIR)/ggml/include -g -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -march=native -fopenmp -std=gnu11 -c $< -o $@ $(OPT)

layer-compute-bench.o: layer-compute-bench.cpp layer-compute-bench.h
	$(CXX) -I$(ROOT_DIR)/src -I$(ROOT_DIR)/ggml/src -I$(ROOT_DIR)/common -I$(ROOT_DIR)/vendor -I$(ROOT_DIR)/include -I$(ROOT_DIR)/ggml/include -I$(ROOT_DIR)/ggml -I$(ROOT_DIR)/ggml/src/ggml-cpu -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -c $< -o $@ $(OPT)

clean:
	rm *.o layer-compute-bench
