import argparse
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
import time
import numpy as np
from pathlib import Path


def compute_answer_prob(scores, answer_ids):
    scores_seq_len = scores.shape[1]
    answer_seq_len = answer_ids.shape[1]

    if answer_seq_len > scores_seq_len:
        pad_size = answer_seq_len - scores_seq_len
        scores = F.pad(scores, (0, 0, 0, pad_size), "constant", 0)
    elif answer_seq_len < scores_seq_len:
        scores = scores[:, :answer_seq_len, :]

    dist = F.softmax(
        scores.masked_fill(scores == -float("inf"), -1e9)
        .gather(2, answer_ids.unsqueeze(-1))
        .squeeze(-1),
        dim=-1,
    )
    return dist


def compute_i2c(args, model, tokenizer, image_tensors, data_samples):
    batch_size = len(data_samples)
    i2c_scores = torch.zeros(batch_size, len(data_samples[0]["conversations"]) // 2)

    input_ids_with_images = []
    input_ids_without_images = []
    answer_ids_list = []

    for idx, data_sample in enumerate(data_samples):
        conversations = data_sample["conversations"]
        for i in range(0, len(conversations), 2):
            question = conversations[i]["value"]
            answer = conversations[i + 1]["value"]

            conv_with_image = conv_templates[args.conv_mode].copy()
            question_with_image = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
            conv_with_image.append_message(
                conv_with_image.roles[0], question_with_image
            )
            conv_with_image.append_message(conv_with_image.roles[1], None)

            conv_without_image = conv_templates[args.conv_mode].copy()
            conv_without_image.append_message(conv_without_image.roles[0], question)
            conv_without_image.append_message(conv_without_image.roles[1], None)

            input_ids_with_images.append(
                tokenizer_image_token(
                    conv_with_image.get_prompt(),
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
            )
            input_ids_without_images.append(
                tokenizer(
                    conv_without_image.get_prompt(), return_tensors="pt"
                ).input_ids[0]
            )
            answer_ids_list.append(tokenizer(answer, return_tensors="pt").input_ids[0])

    # Use pad_sequence to pad the sequences to the same length
    input_ids_with_images = torch.nn.utils.rnn.pad_sequence(
        input_ids_with_images, batch_first=True, padding_value=IGNORE_INDEX
    ).to(args.device)
    input_ids_without_images = torch.nn.utils.rnn.pad_sequence(
        input_ids_without_images, batch_first=True, padding_value=IGNORE_INDEX
    ).to(args.device)
    answer_ids_list = torch.nn.utils.rnn.pad_sequence(
        answer_ids_list, batch_first=True, padding_value=IGNORE_INDEX
    ).to(args.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids_with_images,
            images=image_tensors.half().to(args.device),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            use_cache=True,
        )
        scores = torch.cat(outputs.scores, dim=0).unsqueeze(dim=0)

    s_a_v = compute_answer_prob(scores, answer_ids_list)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids_without_images,
            images=None,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            use_cache=True,
        )
        scores = torch.cat(outputs.scores, dim=0).unsqueeze(dim=0)

    s_a = compute_answer_prob(scores, answer_ids_list)

    kl_div = F.kl_div(
        F.log_softmax(s_a_v, dim=-1), s_a, reduction="batchmean", log_target=False
    )
    i2c_scores[:, :] = kl_div

    return i2c_scores


def I2C_gen(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, load_4bit=args.load_4bit
    )

    with open(os.path.expanduser(args.question_file), "r") as f:
        datasets = f.readlines()

    datasets = [json.loads(line) for line in datasets]

    batch_size = args.batch_size
    i2c_tensor = torch.zeros(len(datasets), len(datasets[0]["conversations"]) // 2)

    for batch_start in tqdm(range(0, len(datasets), batch_size)):
        batch = datasets[batch_start : batch_start + batch_size]
        images = [
            Image.open(
                os.path.join(
                    Path(args.dataset_path).expanduser(),
                    args.dataset_prefix + data["image"],
                )
            )
            for data in batch
        ]
        image_tensors = torch.cat(
            [
                image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                for image in images
            ]
        ).to(args.device)

        i2c_scores = compute_i2c(args, model, tokenizer, image_tensors, batch)
        i2c_tensor[batch_start : batch_start + batch_size] = i2c_scores

    torch.save(i2c_tensor, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--dataset-path", type=str, default="~/fiftyone/coco-2014/train/data"
    )
    parser.add_argument("--dataset-prefix", type=str, default="COCO_train2014_")
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--save-path", type=str, default="./C3L/I2C.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    I2C_gen(args)
