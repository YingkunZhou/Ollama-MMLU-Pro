export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/Magistral-Small-2509-thresholds.txt
export SPARSE_THRESHOLD=models/Magistral-Small-2509-thresholds.txt
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/magistral-system.txt
export SYSF=magistral-system.txt
EXE=./run-speculative-benchmark.sh
QUANT_TYPE=$1

$EXE Magistral-Small-2509 $QUANT_TYPE aime2025 50 0.7 32768 0.95
$EXE Magistral-Small-2509 $QUANT_TYPE gpqa-diamond 50 0.7 32768 0.95
$EXE Magistral-Small-2509 $QUANT_TYPE mmlu-pro-computer_science 50 0.7 32768 0.95
$EXE Magistral-Small-2509 $QUANT_TYPE livecodebench-lite 50 0.7 32768 0.95
