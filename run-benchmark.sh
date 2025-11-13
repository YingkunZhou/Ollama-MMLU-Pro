log_command() {
    local log_file="$1"
    shift
    local cmd="$*"

    echo "\$ $cmd" | tee "$log_file"
    echo "Test Time: $(LC_TIME=en_US.UTF-8 date '+%a %b %d %H:%M:%S %Z %Y')" | tee -a "$log_file"
    eval "$cmd" | tee -a "$log_file"
}

CTX=8192

# Qwen3: For non-thinking mode (enable_thinking=False), we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
MIN_P=0.0
PENALTY=0.2

MODEL_NAME=Qwen3-0.6B-BF16
MODEL=models/$MODEL_NAME.gguf
mkdir -p benchmark/$MODEL_NAME

BENCHMARK_NAME=humaneval
BENCHMARK=benchmark/$BENCHMARK_NAME.txt
DUMPLOG=benchmark/$MODEL_NAME/$BENCHMARK_NAME.log

log_command $DUMPLOG llama.cpp/cuda_build/bin/llama-cli -c $CTX -n $CTX -m $MODEL -bm $BENCHMARK --top-k $TOP_K --temp $TEMPERATURE --top-p $TOP_P --min-p $MIN_P --presence-penalty $PENALTY -ngl 99 -t 1 -fa --seed 42 --no-think " /no_think"