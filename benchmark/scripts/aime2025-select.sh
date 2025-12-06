#!/bin/bash
# run_aime2025_select.sh

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
  -l "results/${model}-${quant}-spec-sparse/aime2025.log" \
  -i "aime2025.txt" \
  -k $k \
  -o "./test_question_select/${model}/${quant}/aime2025_order$k.txt"