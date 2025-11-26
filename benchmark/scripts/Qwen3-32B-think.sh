export SUFFIX="-sparse-think"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/Qwen3-32B-thresholds.txt
export SPARSE_THRESHOLD=models/Qwen3-32B-thresholds.txt
export THINK="\" /think"\"
EXE=./run-speculative-benchmark.sh
QUANT_TYPE=$1

$EXE Qwen3-32B $QUANT_TYPE aime2025 20 0.6 32768 0.95 1.5 0
$EXE Qwen3-32B $QUANT_TYPE gpqa-diamond 20 0.6 32768 0.95 1.5 0
$EXE Qwen3-32B $QUANT_TYPE mmlu-pro-computer_science 20 0.6 32768 0.95 1.5 0
$EXE Qwen3-32B $QUANT_TYPE livecodebench-lite 20 0.6 32768 0.95 1.5 0
