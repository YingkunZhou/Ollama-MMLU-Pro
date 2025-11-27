export SUFFIX="-sparse"
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/Mistral-Small-3.2-24B-Instruct-2506-thresholds.txt
export SPARSE_THRESHOLD=models/Mistral-Small-3.2-24B-Instruct-2506-thresholds.txt
### https://github.com/YingkunZhou/Ollama-MMLU-Pro/releases/download/v0.4/mistral-system.txt
export SYSF=mistral-system.txt
EXE=./run-speculative-benchmark.sh
# export NGLD=99
QUANT_TYPE=$1

$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  aime2025 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  gpqa-diamond 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  humaneval 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  livecodebench-lite 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-pro-computer_science 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-pro-engineering 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-pro-health 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-pro-history 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-pro-law 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-pro-other 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-pro-philosophy 50 0.15
$EXE Mistral-Small-3.2-24B-Instruct-2506 $QUANT_TYPE  mmlu-redux 50 0.15