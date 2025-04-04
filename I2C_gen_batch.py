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
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from PIL import Image
import numpy as np
from pathlib import Path
import csv


def get_chunk(lst, n, k):
    return np.array_split(lst, n)[k]


def compute_answer_prob(scores, answer_ids):
    batch_size, scores_seq_len, vocab_size = scores.shape
    _, answer_seq_len = answer_ids.shape

    if answer_seq_len > scores_seq_len:
        # Pad scores with zeros along the sequence length
        pad_size = answer_seq_len - scores_seq_len
        scores = F.pad(
            scores, (0, 0, 0, pad_size), value=0
        )  # Pad only along seq_len dim
    elif answer_seq_len < scores_seq_len:
        # Truncate scores to match answer_ids length
        scores = scores[:, :answer_seq_len, :]

    # Masking and softmax over vocabulary dimension
    masked_scores = scores.masked_fill(scores == -float("inf"), -1e9)

    # Gather probabilities at answer indices
    gathered_scores = masked_scores.gather(
        2, answer_ids.unsqueeze(-1)
    )  # (batch, seq_len, 1)

    # Remove the last dimension
    dist = F.softmax(gathered_scores.squeeze(-1), dim=-1)  # (batch, seq_len)

    return dist


def compute_i2c(args, model, tokenizer, image_tensor, data_sample):
    conversations = data_sample["conversations"]
    input_ids_with_images = []
    input_ids_without_images = []
    answer_ids = []

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
        conv_with_image.append_message(conv_with_image.roles[0], question_with_image)
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
            tokenizer(conv_without_image.get_prompt(), return_tensors="pt").input_ids[0]
        )
        answer_ids.append(tokenizer(answer, return_tensors="pt").input_ids[0])

    # Use pad_sequence to pad the sequences to the same length
    input_ids_with_images = torch.nn.utils.rnn.pad_sequence(
        input_ids_with_images, batch_first=True, padding_value=IGNORE_INDEX
    ).to(args.device)
    input_ids_without_images = torch.nn.utils.rnn.pad_sequence(
        input_ids_without_images, batch_first=True, padding_value=IGNORE_INDEX
    ).to(args.device)
    answer_ids = torch.nn.utils.rnn.pad_sequence(
        answer_ids, batch_first=True, padding_value=IGNORE_INDEX
    ).to(args.device)
    # Images need to be repeated
    images = (
        image_tensor.unsqueeze(0)
        .half()
        .repeat(len(input_ids_with_images), 1, 1, 1)
        .to(args.device)
    )

    # S(A|V): Visual Answer Scores
    with torch.no_grad():
        outputs = model.generate(
            input_ids_with_images,
            images=images,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            use_cache=True,
        )
        scores = torch.stack(outputs.scores, dim=0).permute(1, 0, 2)
    s_a_v = compute_answer_prob(scores, answer_ids)

    # S(A): Direct Answer Scores (without image)
    with torch.no_grad():
        outputs = model.generate(
            input_ids_with_images,
            images=None,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            use_cache=True,
        )
        scores = torch.stack(outputs.scores, dim=0).permute(1, 0, 2)
    s_a = compute_answer_prob(scores, answer_ids)

    # Compute KL-divergence
    kl_div = F.kl_div(
        F.log_softmax(s_a_v, dim=-1), s_a, reduction="none", log_target=False
    ).sum(dim=-1)

    return kl_div


def I2C_gen(args):
    # Model setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, load_4bit=args.load_4bit
    )

    # Dataset setup
    with open(os.path.expanduser(args.question_file), "r") as f:
        datasets = f.readlines()
    datasets = get_chunk(datasets, args.num_chunks, args.chunk_idx)

    # CSV file path
    csv_file_path = os.path.expanduser(args.save_path)

    # Determine the last processed index by counting existing lines
    last_index = 0
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode="r") as csvfile:
            last_index = sum(1 for _ in csvfile)  # Count number of existing rows

    # Start processing directly from the last index
    for idx in tqdm(
        range(last_index, len(datasets)), initial=last_index, total=len(datasets)
    ):
        data_sample = json.loads(datasets[idx])

        image = os.path.join(
            Path(args.dataset_path).expanduser(),
            args.dataset_prefix + data_sample["image"],
        )
        image = Image.open(image)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # Compute I2C scores
        i2c_scores = compute_i2c(args, model, tokenizer, image_tensor, data_sample)

        # Append I2C scores to CSV
        with open(csv_file_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(i2c_scores.tolist())  # Save only I2C scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument(
        "--dataset-path", type=str, default="~/fiftyone/coco-2014/train/data"
    )
    parser.add_argument("--dataset-prefix", type=str, default="COCO_train2014_")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--save-path", type=str, default="./C3L/I2C.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    I2C_gen(args)
