#!/bin/bash
# run_humaneval_select.sh

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
  -l "results/${model}-${quant}-spec-sparse/humaneval.log" \
  -i "humaneval.txt" \
  -k $k \
  -o "./test_question_select/${model}/${quant}/humaneval_order$k.txt"