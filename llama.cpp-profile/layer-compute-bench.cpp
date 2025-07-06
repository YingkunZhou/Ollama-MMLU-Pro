#include "layer-compute-bench.h"

static bool zyk_collect_activations(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_activations(t, ask, user_data);
}

static void test_layer(llama_context * ctx, std::vector<struct layer_tinfo> &layers,
                           bool use_cpu_backend, bool no_warmup=false) {
    ggml_backend_sched_t sched = ctx->get_sched();

    // static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched)
    // assert the cpu backend split is always the first
    struct ggml_backend_sched_split * split = &sched->splits[0];
    ggml_backend_t backend = sched->backends[split->backend_id];
    ggml_cgraph * cgraph = &split->graph;
    // static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph)
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    assert(cpu_ctx->work_size == cplan.work_size);
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;
    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    if (sched->n_splits == 1) { // only use cpu backend
        for (auto &layer: layers) {
            for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
                struct ggml_tensor * node = cgraph->nodes[node_n];
                struct ggml_tensor * weight = node->src[0];
                if (layer.name == weight->name) {
                    struct Stats stat = g_collector.get_layer(layer.name);
                    struct ggml_tensor * input = node->src[1];
                    assert(stat.input_act.size() == ggml_nbytes(input));
                    memcpy(input->data, stat.input_act.data(), ggml_nbytes(input));

                    layer.samples_ns += layer_cpu_compute(&cplan, node_n, nullptr);

                    assert(stat.output_act.size() == ggml_nbytes(node));
                    assert(floatArraysEqual((float*)stat.output_act.data(), (float*)node->data, ggml_nelements(node)));
                    break;
                }
            }
        }
    }
    else {
        assert(sched->n_copies == 1 && sched->n_backends == 2);
        struct ggml_backend_sched_split * split = &sched->splits[1];
        struct ggml_tensor * src0 = split->inputs[0];
        struct ggml_tensor * src1 = split->inputs[1];
        assert(layers.size() == 1 && layers[0].name == src0->name);
        assert(cgraph->nodes[cgraph->n_nodes-1] == src1);
        struct Stats stat = g_collector.get_layer(src0->name);
        const size_t output_nbytes = ggml_nbytes(split->graph.nodes[0]);
        const size_t output_nelements = ggml_nelements(split->graph.nodes[0]);
        assert(stat.output_act.size() == output_nbytes);
        stat.n_params = src0->ne[0]*src0->ne[1];
        stat.size = src0->nb[2];
        std::vector<char> result;
        result.resize(output_nbytes);
        const char* env_epsilon = std::getenv("EPSILON");
        float epsilon = env_epsilon? std::atof(env_epsilon) : 1e-6f;

        if (use_cpu_backend) {
            struct ggml_tensor * output = ggml_mul_mat(ctx->get_ctx_compute(), src0, src1);
            output->data = (void *) result.data();

            cplan.work_size = MAX(cplan.work_size, ggml_row_size(GGML_TYPE_Q8_K, ggml_nelements(src1)));
            if (cpu_ctx->work_size < cplan.work_size) {
                delete[] cpu_ctx->work_data;
                cpu_ctx->work_data = new uint8_t[cplan.work_size];
                cpu_ctx->work_size = cplan.work_size;
            }
            cplan.work_data = (uint8_t *)cpu_ctx->work_data;

            layer_cpu_compute(&cplan, -1, output);

            if (no_warmup) { // validate the output result
                assert(floatArraysEqual((float*)stat.output_act.data(), (float*)output->data, output_nelements, epsilon));
                return;
            }
            memcpy(stat.output_act.data(), output->data, output_nbytes);
        }
        else {
            int split_backend_id = split->backend_id;
            ggml_backend_t split_backend = sched->backends[split_backend_id];
            struct ggml_tensor * src0_cpy = sched->hv_tensor_copies[2*ggml_hash_find(&sched->hash_set, src0)+split_backend_id];
            struct ggml_tensor * src1_cpy = sched->hv_tensor_copies[2*ggml_hash_find(&sched->hash_set, src1)+split_backend_id];

            layers[0].samples_ns += layer_gpu_compute(src0, src1, src0_cpy, src1_cpy, split->graph.nodes[0], split_backend->context);

            ggml_backend_tensor_get(split->graph.nodes[0], result.data(), 0, output_nbytes);
            assert(floatArraysEqual((float*)stat.output_act.data(), (float*)split->graph.nodes[0]->data, output_nelements, epsilon));
        }
    }
}

static void test_gen(llama_context * ctx, int n_gen,
    uint64_t &samples_ns, std::vector<struct layer_tinfo> &layers) {
    // very important, otherwise will fault
    llama_memory_clear(llama_get_memory(ctx), false);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

    uint64_t t_start;
    for (int i = 0; i < n_gen; i++) {
        t_start = get_time_ns();
        llama_batch batch = llama_batch_get_one(&token, 1);

        const int ret = ctx->decode(batch);

        assert(ret == 0); // line 12
        samples_ns += get_time_ns() - t_start;

        test_layer(ctx, layers, false);

        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
    }
    for (auto &layer: layers) {
        layer.ts = 1e9 * n_gen / layer.samples_ns;
    }
}

int main(int argc, char ** argv) {
    // try to set locale for unicode characters in markdown
    setlocale(LC_CTYPE, ".UTF-8");
    // initialize backends
    ggml_backend_load_all();

    cmd_params params = parse_cmd_params(argc, argv);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");

    // initialize llama.cpp
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    set_process_priority(params.prio);

    // initialize printer
    std::unique_ptr<printer> p = std::unique_ptr<printer>(new markdown_printer());
    p->fout = stdout;
    p->print_header(params);

    std::vector<cmd_params_instance> params_instances = get_cmd_params_instances(params);

    auto inst = params_instances[0];
    // construct llama_model instance from static gguf file
    llama_model * lmodel = llama_model_load_from_file(inst.model.c_str(), inst.to_llama_mparams());
    // construct llama_context instance for dynamic running environment
    llama_context * ctx = llama_init_from_model(lmodel, inst.to_llama_cparams());
    test t(inst, lmodel, ctx);
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(t.n_threads);
    parse_cpu_mask(t.cpu_mask, tpp.cpumask);
    tpp.strict_cpu = t.cpu_strict;
    tpp.poll       = t.poll;
    tpp.prio       = params.prio;
    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    llama_attach_threadpool(ctx, threadpool, NULL);
    llama_memory_clear(llama_get_memory(ctx), false);
    llama_set_n_threads(ctx, t.n_threads, t.n_threads); // TODO: maybe pp and tg can be different

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    ggml_backend_sched_t sched  = ctx->get_sched();
    g_collector.set_layers(inst.layers);
    if (t.n_gen > 0) {
        llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        llama_batch batch = llama_batch_get_one(&token, 1);
        // if whole model is running on cpu, or gpu without warmup, which means just run cpu-gpu result checkout
        if (model->dev_layer(0) == cpu_dev || params.no_warmup) {
            sched->is_alloc = true;
            sched->callback_eval = zyk_collect_activations;
        }
        // warmup
        const int ret = ctx->decode(batch);
        assert(ret == 0);
        // if model is running on gpu
        if (model->dev_layer(0) != cpu_dev) {
            test_layer(ctx, t.layers, true, params.no_warmup);
            // if just checkout the CPU and GPU output result
            if (params.no_warmup) return 0;
        }
        // begin testing
        test_gen(ctx, t.n_gen, t.samples_ns, t.layers);
    }

    p->print_test(t);
    fflush(p->fout);
    llama_perf_context_print(ctx);

    llama_free(ctx);

    ggml_threadpool_free_fn(threadpool);
    llama_model_free(lmodel);
    llama_backend_free();
    return 0;
}
