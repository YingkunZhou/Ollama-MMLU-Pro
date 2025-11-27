export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/Qwen3-32B-thresholds.txt
export SPARSE_THRESHOLD=models/Qwen3-32B-thresholds.txt
export THINK="\" /no_think"\"
EXE=./run-speculative-benchmark.sh
# export NGLD=99
QUANT_TYPE=$1

$EXE Qwen3-32B $QUANT_TYPE  aime2025 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  gpqa-diamond 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  livecodebench-lite 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  humaneval 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-pro-computer_science 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-pro-engineering 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-pro-health 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-pro-history 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-pro-law 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-pro-other 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-pro-philosophy 20 0.7 8192 0.8 1.5 0
$EXE Qwen3-32B $QUANT_TYPE  mmlu-redux 20 0.7 8192 0.8 1.5 0