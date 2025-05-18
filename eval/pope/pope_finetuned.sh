#!/bin/bash

MODEL_NAME="llava-v1.5-7b-finetuned-lora"
BASE_MODEL_NAME="llava-v1.5-7b"

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$MODEL_NAME \
    --model-base /home/matthewwang16czap/models/$BASE_MODEL_NAME \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/matthewwang16czap/fiftyone/coco-2014/validation/data \
    --answers-file ./playground/data/eval/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode llava_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$MODEL_NAME.jsonl \
    > ./C3L/eval/pope/results/pope_$MODEL_NAME.txt 2>&1
