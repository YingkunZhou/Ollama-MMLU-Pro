# usage

```bash
export ROOT_DIR=<your llama.cpp path>/llama.cpp
export BUILD_DIR=<your build folder in llama.cpp>
export LD_LIBRARY_PATH=${ROOT_DIR}/${BUILD_DIR}/bin

make clean
make
# OPT="-g" make
./layer-compute-bench -m <your model path>/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf -l blk.0.attn_q.weight -p 0 -n 64 -t 8 -ngl 3
```

## sample result

| model                          |         size |       params | backend    | threads |            test |                  t/s |
| ------------------------------ | -----------: | -----------: | ---------- | ------: | --------------: | -------------------: |
| llama 8B Q2_K - Medium         |  2513.80 MiB |    8030.26 M | CPU        |       8 |            tg64 |                 23.80 |
| blk.0.attn_q.weight            |     5.25 MiB |      16.78 M | CPU        |       8 |            tg64 |              11481.10 |