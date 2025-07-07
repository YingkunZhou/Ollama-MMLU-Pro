# usage

```bash
export ROOT_DIR=${PWD}
export LLAMA_CPP_DIR=<your llama.cpp path>/llama.cpp # e.g. /tmp/llama.cpp
export BUILD_DIR=<your cpu build folder in llama.cpp> # e.g. cpu_build
```

## build llama.cpp with cuda backend

### GPU

```bash
cd ${LLAMA_CPP_DIR}
patch -p1 < ${ROOT_DIR}/gpu-profile.patch
cmake -B ${BUILD_DIR} -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DLLAMA_CURL=OFF #-DCMAKE_BUILD_TYPE=Debug
cmake --build ${BUILD_DIR} --config Release -j $(nproc)
```


### CPU

```bash
cd ${LLAMA_CPP_DIR}
cmake -B cpu_build -DGGML_CUDA=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DLLAMA_CURL=OFF #-DCMAKE_BUILD_TYPE=Debug
cmake --build cpu_build --config Release -j $(nproc)
```

## build and run profiling test

### GPU

```bash
make clean
make layer-gpu-bench
# make
# OPT="-g" make
export LD_LIBRARY_PATH=${ROOT_DIR}/${BUILD_DIR}/bin
EPSILON=0.032 ./layer-gpu-bench -m <your model path>/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf -l blk.0.attn_q.weight -p 0 -n 128 -t 1
```

| model                          |         size |       params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | -----------: | -----------: | ---------- | --: | ------: | --------------: | -------------------: |
| llama 8B Q2_K - Medium         |  2513.80 MiB |    8030.26 M | CUDA       |  99 |       1 |           tg128 |                179.56 |
| blk.0.attn_q.weight            |     5.25 MiB |      16.78 M | CUDA       |  99 |       1 |           tg128 |              59445.08 |


### CPU

```bash
make clean
NOGPU="-DNOGPU" make layer-cpu-bench
# OPT="-g" NOGPU="-DNOGPU" make layer-cpu-bench
export LD_LIBRARY_PATH=${ROOT_DIR}/cpu_build/bin
./layer-cpu-bench -m <your model path>/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf -l blk.0.attn_q.weight -p 0 -n 64 -t 8 -ngl 0 --no-warmup
```

| model                          |         size |       params | backend    | threads |            test |                  t/s |
| ------------------------------ | -----------: | -----------: | ---------- | ------: | --------------: | -------------------: |
| llama 8B Q2_K - Medium         |  2513.80 MiB |    8030.26 M | CPU        |       8 |            tg64 |                 23.86 |
| blk.0.attn_q.weight            |     5.25 MiB |      16.78 M | CPU        |       8 |            tg64 |              11658.04 |