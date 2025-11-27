export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/phi-4-thresholds.txt
export SPARSE_THRESHOLD=models/phi-4-thresholds.txt
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/phi4-system.txt
export SYSF=phi4-system.txt
# export NGLD=99
EXE=./run-speculative-benchmark.sh
QUANT_TYPE=$1

$EXE phi-4 $QUANT_TYPE  aime2025 50 0.5
$EXE phi-4 $QUANT_TYPE  gpqa-diamond 50 0.5
$EXE phi-4 $QUANT_TYPE  humaneval 50 0.5
$EXE phi-4 $QUANT_TYPE  livecodebench-lite 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-pro-computer_science 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-pro-engineering 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-pro-health 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-pro-history 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-pro-law 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-pro-other 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-pro-philosophy 50 0.5
$EXE phi-4 $QUANT_TYPE  mmlu-redux 50 0.5