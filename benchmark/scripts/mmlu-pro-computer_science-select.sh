#!/bin/bash
# run_mmlu-pro-computer_science_select.sh

model="Qwen3-32B"
quant="exl3-2.0bpw"
k=10

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
  -i "mmlu-pro-computer_science.txt" \
  -k $k \
  -o "./test_question_select/${model}/${quant}/mmlu-pro-computer_science_order$k.txt"