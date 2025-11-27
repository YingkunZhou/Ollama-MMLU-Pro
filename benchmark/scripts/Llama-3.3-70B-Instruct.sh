export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/Llama-3.3-70B-Instruct-thresholds.txt
export SPARSE_THRESHOLD=models/Llama-3.3-70B-Instruct-thresholds.txt
EXE=./run-speculative-benchmark.sh
# export NGLD=99
QUANT_TYPE=$1

$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  aime2025 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  gpqa-diamond 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  humaneval 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  livecodebench-lite 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-pro-computer_science 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-pro-engineering 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-pro-health 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-pro-history 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-pro-law 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-pro-other 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-pro-philosophy 50 0.6 8192 0.9
$EXE Llama-3.3-70B-Instruct $QUANT_TYPE  mmlu-redux 50 0.6 8192 0.9
