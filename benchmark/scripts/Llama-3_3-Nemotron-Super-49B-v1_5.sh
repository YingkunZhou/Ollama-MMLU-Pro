export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/Llama-3_3-Nemotron-Super-49B-v1_5-thresholds.txt
export SPARSE_THRESHOLD=models/Llama-3_3-Nemotron-Super-49B-v1_5-thresholds.txt
EXE=./run-speculative-benchmark.sh
QUANT_TYPE=$1

$EXE Llama-3_3-Nemotron-Super-49B-v1_5 $QUANT_TYPE aime2025 50 0.6 32768 0.95
$EXE Llama-3_3-Nemotron-Super-49B-v1_5 $QUANT_TYPE gpqa-diamond 50 0.6 32768 0.95
$EXE Llama-3_3-Nemotron-Super-49B-v1_5 $QUANT_TYPE mmlu-pro-computer_science 50 0.6 32768 0.95
$EXE Llama-3_3-Nemotron-Super-49B-v1_5 $QUANT_TYPE livecodebench-lite 50 0.6 32768 0.95
