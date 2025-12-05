#!/bin/bash

model="Qwen3-32B"
quant="exl3-2.25bpw"
k=2

while getopts "m:q:k:" opt; do
    case $opt in
        m) model="$OPTARG" ;;
        q) quant="$OPTARG" ;;
        k) k="$OPTARG" ;;
        *) exit 1 ;;
    esac
done

python question-select.py \
  -l "paper2-results/${model}-${quant}-spec-sparse/mmlu-pro-computer_science.log" \
     "paper2-results/${model}-${quant}-spec-sparse/mmlu-pro-engineering.log" \
     "paper2-results/${model}-${quant}-spec-sparse/mmlu-pro-health.log" \
     "paper2-results/${model}-${quant}-spec-sparse/mmlu-pro-history.log" \
     "paper2-results/${model}-${quant}-spec-sparse/mmlu-pro-law.log" \
     "paper2-results/${model}-${quant}-spec-sparse/mmlu-pro-other.log" \
     "paper2-results/${model}-${quant}-spec-sparse/mmlu-pro-philosophy.log" \
  -i "mmlu-pro-computer_science.txt" \
     "mmlu-pro-engineering.txt" \
     "mmlu-pro-health.txt" \
     "mmlu-pro-history.txt" \
     "mmlu-pro-law.txt" \
     "mmlu-pro-other.txt" \
     "mmlu-pro-philosophy.txt" \
  -k $k \
  -o "./test_question_select/${model}/${quant}/mmlu-pro-order$k.txt"