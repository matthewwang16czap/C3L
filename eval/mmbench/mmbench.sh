#!/bin/bash

SPLIT="mmbench_dev_20230712"
MODEL_NAME="llava-v1.5-7b"

python -m llava.eval.model_vqa_mmbench \
    --model-path /home/matthew/models/$MODEL_NAME \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llava_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $MODEL_NAME 

python ./C3L/eval/mmbench/mmbench_get_accuracy.py \
    --file-path ./playground/data/eval/mmbench/answers_upload/$SPLIT/$MODEL_NAME.xlsx  \
    > ./C3L/eval/mmbench/results/mmbench_$MODEL_NAME.txt 2>&1
