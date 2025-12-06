#!/bin/bash
# run_gpqa-diamond_select.sh

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
  -l "results/${model}-${quant}-spec-sparse/gpqa-diamond.log" \
  -i "gpqa-diamond.txt" \
  -k $k \
  -o "./test_question_select/${model}/${quant}/gpqa-diamond_order$k.txt"