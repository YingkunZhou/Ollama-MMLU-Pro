#!/usr/bin/env bash

log_command() {
    local log_file="$1"
    shift
    local cmd="$*"

    echo "\$ $cmd" | tee "$log_file"
    echo "Test Time: $(LC_TIME=en_US.UTF-8 date '+%a %b %d %H:%M:%S %Z %Y')" | tee -a "$log_file"
    eval "$cmd" | tee -a "$log_file"
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
RDMODEL="models/${MODEL_NAME}-${QUANT_TYPE}-residual-IQ2_KS.gguf"
BENCHMARK_NAME=${3:-"humaneval"}
BENCHMARK="benchmark/${BENCHMARK_NAME}.txt"

# ----------- output log -----------
SUFFIX=${SUFFIX:-""}
OUT_DIR="benchmark/${MODEL_NAME}-${QUANT_TYPE}-spec${SUFFIX}"
mkdir -p "$OUT_DIR"

TOP_K=${4:-1}
TEMPERATURE=${5:-0.0}
CTX=${6:-8192}
TOP_P=${7:-1.0}
PENALTY=${8:-0.0}
MIN_P=${9:-0.0}

SAMPLING_FLAG=""
if [ "$TOP_K" -eq 1 ]; then
    SAMPLING_FLAG="--sampling-seq k"
fi

SYSF_FLAG=""
if [ -n "$SYSF" ]; then
    SYSF_FLAG="-sysf $SYSF"
fi

THINK_FLAG=""
if [ -n "$THINK" ]; then
    THINK_FLAG="-tk $THINK"
fi

SPARSE_THRESHOLD=""
if [ -n "$SPARSE_THRESHOLD" ]; then
    SPARSE_FLAG="--threshold $SPARSE_THRESHOLD"
fi

# ----------- benchmarking -----------
DUMPLOG="${OUT_DIR}/${BENCHMARK_NAME}.log"

echo "Running benchmark: ${BENCHMARK_NAME}"

log_command $DUMPLOG \
    llama.cpp/cuda_build/bin/llama-speculative \
        -c $CTX -n $CTX \
        -m $MODEL -md $RDMODEL \
        -bm $BENCHMARK \
        --top-k $TOP_K \
        --top-p $TOP_P \
        --min-p $MIN_P \
        --temp $TEMPERATURE \
        --presence-penalty $PENALTY \
        -ngl 99 -ngld 99 -t 8 -fa --seed 42 \
        --draft-max 4 --draft-min 4 --draft-p-min 0.0 \
        $SAMPLING_FLAG $SYSF_FLAG $THINK_FLAG $SPARSE_FLAG

### command example
# please refer to run-benchmark.sh
