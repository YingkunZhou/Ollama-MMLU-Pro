#include "common.cuh"

#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define VDR_Q2_K_Q8_1_MMVQ 1

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts); // 或者 CLOCK_MONOTONIC
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t layer_gpu_compute(ggml_tensor * src0_cpu, ggml_tensor * src1_cpu, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, void * context);
