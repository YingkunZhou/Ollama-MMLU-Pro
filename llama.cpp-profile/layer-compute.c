#include "layer-compute.h"
#if defined __AVX2__
#include "avx2-op.h"
#endif

static void Q2_K_weight_gemv(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS
    assert(ne1 == 1 && ne11 == 1);
    assert(ne0 == ne01 && ne00 == ne10);

    const int ith = params->ith;
    const int nth = params->nth;

    quantize_row_q8_K(
        (float *)((char *)src1->data + ith*nb11/nth),
        (void *) ((char *)params->wdata + ith*ne10*ggml_type_size(GGML_TYPE_Q8_K)/nth/ggml_blck_size(GGML_TYPE_Q8_K)),
        ne10/nth);
    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }
    ggml_barrier(params->threadpool);

    const int chunk_size = 64;
    assert((ne0 / chunk_size) * chunk_size == ne0);
    // attempt to reduce false-sharing (does not seem to make a difference)
    float tmp[chunk_size];
    int64_t nchunk = ne0 / chunk_size;
    assert(nchunk >= nth);

    const char * src0_row = (const char*)src0->data;
    const char * src1_col = (const char*)params->wdata;
    float * dst_col = (float*)dst->data;

    int current_chunk = ith;
    while (current_chunk < nchunk) {
        const int64_t ir0_start = chunk_size * current_chunk;
        for (int64_t ir0 = ir0_start; ir0 < ir0_start + chunk_size; ir0++) {
            vec_dot_q2_K_q8_K(ne00, &tmp[ir0-ir0_start], src0_row + ir0*nb01, src1_col);
        }
        memcpy(&dst_col[ir0_start], tmp, chunk_size * sizeof(float));
        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

static void layer_weight_gemv_test(void * data, int node_n, struct ggml_tensor * node) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };
    if (node) {
        Q2_K_weight_gemv(&params, node);
    }
    else {
        Q2_K_weight_gemv(&params, cgraph->nodes[node_n]);
    }
    assert(!cplan->abort_callback);
    ggml_barrier(state->threadpool);
}

void layer_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan, int node_n, struct ggml_tensor * node) {
    int n_threads = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;
    // Reset some of the parameters that need resetting
    // No worker threads should be accessing the parameters below at this stage
    threadpool->cgraph           = cgraph;
    threadpool->cplan            = cplan;
    threadpool->current_chunk    = 0;
    threadpool->abort            = -1;
    threadpool->ec               = GGML_STATUS_SUCCESS;
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            layer_weight_gemv_test(&threadpool->workers[omp_get_thread_num()], node_n, node);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        layer_weight_gemv_test(&threadpool->workers[0], node_n, node);
    }
    assert(threadpool->ec == GGML_STATUS_SUCCESS);
}