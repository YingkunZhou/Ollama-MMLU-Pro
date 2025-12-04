#!/usr/bin/env bash

log_command() {
    local log_file="$1"
    shift
    local cmd="$*"

    echo "\$ $cmd" | tee "$log_file"
    echo "Test Time begin: $(LC_TIME=en_US.UTF-8 date '+%a %b %d %H:%M:%S %Z %Y')" | tee -a "$log_file"
    eval "$cmd" | tee -a "$log_file"
    echo "Test Time end: $(LC_TIME=en_US.UTF-8 date '+%a %b %d %H:%M:%S %Z %Y')" | tee -a "$log_file"
}

# ----------- parameter process -----------
echo "Usage: $0 <MODEL_NAME> <QUANT_TYPE> <BENCHMARK_NAME> <TOP_K> <TEMPERATURE> <CTX> <TOP_P> <PENALTY> <MIN_P>"

# ----------- benchmark list -----------
BENCHMARK_LIST=(
    aime2025
    # arc-challenge
    gpqa-diamond
    # gsm8k
    humaneval
    livecodebench-lite
    mmlu-pro-biology
    mmlu-pro-business
    mmlu-pro-chemistry
    mmlu-pro-computer_science
    mmlu-pro-economics
    mmlu-pro-engineering
    mmlu-pro-health
    mmlu-pro-history
    mmlu-pro-law
    mmlu-pro-math
    mmlu-pro-other
    mmlu-pro-philosophy
    mmlu-pro-physics
    mmlu-pro-psychology
    mmlu-redux
)

# for BENCHMARK_NAME in "${BENCHMARK_LIST[@]}"; do
# done

MODEL_NAME=${1:-"phi-4"}
QUANT_TYPE=${2:-"Q4_K_M"}
MODEL="models/${MODEL_NAME}-${QUANT_TYPE}.gguf"
BENCHMARK_NAME=${3:-"humaneval"}
BENCHMARK="benchmark/${BENCHMARK_NAME}.txt"

# ----------- output log -----------
SUFFIX=${SUFFIX:-""}
OUT_DIR="benchmark/${MODEL_NAME}-${QUANT_TYPE}${SUFFIX}"
mkdir -p "$OUT_DIR"

TOP_K=${4:-1}
TEMPERATURE=${5:-0.0}
CTX=${6:-8192}
TOP_P=${7:-1.0}
PENALTY=${8:-0.0}
# consistent with llama.cpp default setting
# MIN_P=${9:-0.05}
MIN_P=${9:-0.0}

SAMPLING_FLAG=""
if [ "$TOP_K" -eq 1 ]; then
    SAMPLING_FLAG="--sampling-seq k"
fi

SYSF_FLAG=""
if [ -n "$SYSF" ]; then
    SYSF_FLAG="-sysf $SYSF"
fi

SYSP_FLAG=""
if [ -n "$SYSP" ]; then
    SYSP_FLAG="-sys $SYSP"
fi

THINK_FLAG=""
if [ -n "$THINK" ]; then
    THINK_FLAG="-tk $THINK"
fi

NGL_FLAG="-ngl 99"
if [ -n "$NGL" ]; then
    NGL_FLAG="-ngl $NGL"
fi

KV_QUANT=""
if [ -n "$KV_TYPE" ]; then
    KV_QUANT="-ctk $KV_TYPE -ctv $KV_TYPE"
fi

# ----------- benchmarking -----------
DUMPLOG="${OUT_DIR}/${BENCHMARK_NAME}.log"

echo "Running benchmark: ${BENCHMARK_NAME}"

log_command $DUMPLOG \
    llama.cpp/cuda_build/bin/llama-cli \
        -c $CTX -n $CTX \
        -m $MODEL \
        -bm $BENCHMARK \
        --top-k $TOP_K \
        --top-p $TOP_P \
        --min-p $MIN_P \
        --temp $TEMPERATURE \
        --presence-penalty $PENALTY \
        $NGL_FLAG -t 8 -fa --seed 42 \
        $SAMPLING_FLAG $SYSF_FLAG $SYSP_FLAG $THINK_FLAG $KV_QUANT

### command example
# CUDA_VISIBLE_DEVICES=0
# SUFFIX="-greedy" ./run-benchmark.sh phi-4 BF16 humaneval

### [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905)
### https://huggingface.co/microsoft/phi-4#input-formats
#No.1 SYSF=phi4-system.txt ./run-benchmark.sh phi-4 BF16 humaneval 50 0.5

### https://huggingface.co/microsoft/Phi-4-reasoning-plus#usage
### https://huggingface.co/microsoft/Phi-4-reasoning-plus#input-formats
#No.2 SYSF=phi4rp-system.txt ./run-benchmark.sh Phi-4-reasoning-plus BF16 aime2025 50 0.8 32768 0.95

### https://huggingface.co/Qwen/Qwen3-32B-GGUF#best-practices
### https://huggingface.co/Qwen/Qwen3-32B-GGUF#switching-between-thinking-and-non-thinking-mode
#No.3 THINK="\" /no_think"\" ./run-benchmark.sh Qwen3-32B BF16 humaneval 20 0.7 8192 0.8 1.5 0
#No.4 SUFFIX="-think" THINK="\" /think"\" ./run-benchmark.sh Qwen3-32B BF16 aime2025 20 0.6 32768 0.95 1.5 0

### https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506#usage
### https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/blob/main/SYSTEM_PROMPT.txt
#No.5 SYSF=mistral-system.txt ./run-benchmark.sh Mistral-Small-3.2-24B-Instruct-2506 BF16 humaneval 50 0.15

### https://huggingface.co/mistralai/Magistral-Small-2509#sampling-parameters
### https://huggingface.co/mistralai/Magistral-Small-2509/blob/main/SYSTEM_PROMPT.txt
#No.6 SYSF=magistral-system.txt ./run-benchmark.sh Magistral-Small-2509 BF16 aime2025 50 0.7 32768 0.95

### https://www.reddit.com/r/LocalLLaMA/comments/1j9hsfc/gemma_3_ggufs_recommended_settings/
### https://huggingface.co/google/gemma-3-27b-it/blob/main/generation_config.json
### https://docs.unsloth.ai/models/gemma-3-how-to-run-and-fine-tune
#No.7 ./run-benchmark.sh gemma-3-27b-it BF16 humaneval 64 1.0 8192 0.95

### https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/generation_config.json
#No.8 ./run-benchmark.sh Llama-3.3-70B-Instruct Q4_K_M humaneval 50 0.6 8192 0.9

### https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5#quick-start-and-usage-recommendations
#No.9 ./run-benchmark.sh Llama-3_3-Nemotron-Super-49B-v1_5 Q4_K_M aime2025 50 0.6 32768 0.95
