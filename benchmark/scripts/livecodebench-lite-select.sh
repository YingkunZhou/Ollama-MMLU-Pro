#!/bin/bash
# run_livecodebench-lite_select.sh

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
  -l "paper2-results/${model}-${quant}-spec-sparse/livecodebench-lite.log" \
  -i "livecodebench-lite.txt" \
  -k $k \
  -o "./test_question_select/${model}/${quant}/livecodebench-lite_order$k.txt"