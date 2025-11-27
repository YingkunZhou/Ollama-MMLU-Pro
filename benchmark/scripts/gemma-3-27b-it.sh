export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/gemma-3-27b-it-thresholds.txt
export SPARSE_THRESHOLD=models/gemma-3-27b-it-thresholds.txt
# export NGLD=99
EXE=./run-speculative-benchmark.sh
QUANT_TYPE=$1

$EXE gemma-3-27b-it $QUANT_TYPE aime2025 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE gpqa-diamond 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE humaneval 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE livecodebench-lite 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-pro-computer_science 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-pro-engineering 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-pro-health 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-pro-history 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-pro-law 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-pro-other 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-pro-philosophy 64 1.0 8192 0.95
$EXE gemma-3-27b-it $QUANT_TYPE mmlu-redux 64 1.0 8192 0.95
