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
import math
import time

start_time = time.time()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def compute_answer_prob(scores, answer_ids, tokenizer):
    # Get sequence lengths
    scores_seq_len = scores.shape[1]
    answer_seq_len = answer_ids.shape[1]

    if answer_seq_len > scores_seq_len:
        # Pad probs with zeros along sequence length
        pad_size = answer_seq_len - scores_seq_len
        scores = F.pad(scores, (0, 0, 0, pad_size), "constant", 0)
    elif answer_seq_len < scores_seq_len:
        # Truncate probs to match answer_ids length
        scores = scores[:, :answer_seq_len, :]

    dist = F.softmax(
        scores.masked_fill(
            scores == -float("inf"), -1e9
        )  # Mask for handling -inf values
        .gather(2, answer_ids.unsqueeze(-1))  # Get correct position's score
        .squeeze(-1),
        dim=-1,
    )
    return dist


def compute_i2c(args, model, tokenizer, image_tensor, data_sample, device="cuda"):
    conversations = data_sample["conversations"]

    # initializing
    i2c_scores = torch.zeros(1, len(conversations) // 2)

    for i in range(0, len(conversations), 2):  # USER and ASSISTANT pairs
        question = conversations[i]["value"]
        answer = conversations[i + 1]["value"]

        # consturct prompt
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

        # Tokenize question and answer
        input_ids_with_image = (
            tokenizer_image_token(
                conv_with_image.get_prompt(),
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(device)
        )
        input_ids_without_image = tokenizer(
            conv_without_image.get_prompt(), return_tensors="pt"
        ).input_ids.to(device)
        answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)

        # S(A|V): Visual Answer Scores
        with torch.inference_mode():
            outputs = model.generate(
                input_ids_with_image,
                images=image_tensor.unsqueeze(0).half().cuda(),
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
            )
            scores = torch.concat(outputs.scores, dim=0).unsqueeze(dim=0)
        s_a_v = compute_answer_prob(scores, answer_ids, tokenizer)

        # S(A): Direct Answer Scores (without image)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids_without_image,
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
            scores = torch.concat(outputs.scores, dim=0).unsqueeze(dim=0)
        s_a = compute_answer_prob(scores, answer_ids, tokenizer)

        # Compute KL-divergence
        kl_div = F.kl_div(
            F.log_softmax(s_a_v, dim=-1), s_a, reduction="batchmean", log_target=False
        )
        i2c_scores[0, i // 2] = kl_div

    return i2c_scores


def I2C_gen(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, load_4bit=args.load_4bit
    )

    # Dataset
    with open(os.path.expanduser(args.question_file), "r") as f:
        datasets = f.readlines()
    datasets = get_chunk(datasets, args.num_chunks, args.chunk_idx)

    line_index = 0
    i2c_tensor = None
    for line in tqdm(datasets):
        data_sample = json.loads(line)
        if i2c_tensor is None:
            i2c_tensor = torch.zeros(
                len(datasets), len(data_sample["conversations"]) // 2
            )

        image = os.path.join(
            args.dataset_path, args.dataset_prefix + data_sample["image"]
        )
        image = Image.open(image)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        i2c_scores = compute_i2c(args, model, tokenizer, image_tensor, data_sample)
        i2c_tensor[line_index] = i2c_scores

        line_index = line_index + 1

    torch.save(i2c_tensor, args.save_path)

    print("Time Used:", (time.time() - start_time) / 60, "(min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument("--dataset-path", type=str, default="./dataset/data")
    parser.add_argument("--dataset-prefix", type=str, default="COCO_train2014_")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", type=bool, default=True)
    parser.add_argument("--save-path", type=str, default="./C3L/I2C.pt")
    args = parser.parse_args()

    I2C_gen(args)
