export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/Phi-4-reasoning-plus-thresholds.txt
export SPARSE_THRESHOLD=models/Phi-4-reasoning-plus-thresholds.txt
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/phi4rp-system.txt
export SYSF=phi4rp-system.txt
EXE=./run-speculative-benchmark.sh
QUANT_TYPE=$1

$EXE Phi-4-reasoning-plus $QUANT_TYPE aime2025 50 0.8 32768 0.95
$EXE Phi-4-reasoning-plus $QUANT_TYPE gpqa-diamond 50 0.8 32768 0.95
$EXE Phi-4-reasoning-plus $QUANT_TYPE mmlu-pro-computer_science 50 0.8 32768 0.95
$EXE Phi-4-reasoning-plus $QUANT_TYPE livecodebench-lite 50 0.8 32768 0.95
