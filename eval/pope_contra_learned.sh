#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-contrastive-learned\
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/matthew/fiftyone/coco-2014/validation/data \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-contra-learned.jsonl \
    --temperature 0 \
    --conv-mode llava_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-contra-learned.jsonl
