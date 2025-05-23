import argparse
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from llava.conversation import conv_templates
import csv
from utils import (
    get_chunk,
    tokenize_input,
    load_pretrained_model_from_path,
    get_image_tensors,
)


def score_with_gaussian_weighting(scores, answer_ids, pad_token_id=None, sigma=1.0):
    """
    scores: Tensor (B, T_pred, V)
    answer_ids: Tensor (B, T_true)
    Returns: Tensor (B, T_true) of weighted probabilities per token
    """
    B, T_pred, V = scores.shape
    _, T_true = answer_ids.shape

    device = scores.device

    # (B, T_pred, V) → (B, T_pred, V) log-probabilities
    log_probs = F.log_softmax(scores, dim=-1)

    # === Gather log probs of the correct token across all predicted positions ===
    # (B, T_true) → (B, 1, T_true) → (B, T_pred, T_true)
    index = answer_ids.unsqueeze(1).expand(B, T_pred, T_true)
    token_log_probs = log_probs.gather(2, index)  # (B, T_pred, T_true)

    # === Compute Gaussian weights centered at true positions ===
    # (T_pred,) - all predicted positions
    pred_pos = torch.arange(T_pred, device=device).view(1, T_pred, 1)  # (1, T_pred, 1)
    true_pos = torch.arange(T_true, device=device).view(1, 1, T_true)  # (1, 1, T_true)

    # (1, T_pred, T_true)
    weights = torch.exp(-((pred_pos - true_pos) ** 2) / (2 * sigma**2))
    # weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize over T_pred

    # === Mask out padding tokens if needed ===
    if pad_token_id is not None:
        pad_mask = (answer_ids == pad_token_id).view(B, 1, T_true)  # (B, 1, T_true)
        token_log_probs = token_log_probs.masked_fill(pad_mask, float("-inf"))

    # === Compute weighted sum of probabilities for each token ===
    weighted_probs = (token_log_probs.exp() * weights).sum(dim=1)  # (B, T_true)

    return weighted_probs  # You can also .mean(dim=1) to get per-sentence score


def compute_i2c(args, model, tokenizer, image_tensors, batch_data, pad_token_id):
    s_a_vs = []
    s_as = []

    for i in range(0, args.questions_num * 2, 2):
        questions = [data["conversations"][i]["value"] for data in batch_data]
        answers = [data["conversations"][i + 1]["value"] for data in batch_data]

        input_ids_with_images = [
            tokenize_input(
                question,
                True,
                tokenizer,
                model.config.mm_use_im_start_end,
                conv_templates[args.conv_mode].copy(),
            )
            for question in questions
        ]

        input_ids_without_images = [
            tokenize_input(
                question,
                False,
                tokenizer,
                model.config.mm_use_im_start_end,
                conv_templates[args.conv_mode].copy(),
            )
            for question in questions
        ]

        answer_ids = [
            tokenize_input(
                answer,
                False,
                tokenizer,
                model.config.mm_use_im_start_end,
                None,
            )
            for answer in answers
        ]

        # Use pad_sequence to pad the sequences to the same length
        input_ids_with_images = torch.nn.utils.rnn.pad_sequence(
            input_ids_with_images, batch_first=True, padding_value=pad_token_id
        )

        input_ids_without_images = torch.nn.utils.rnn.pad_sequence(
            input_ids_without_images, batch_first=True, padding_value=pad_token_id
        )

        answer_ids = torch.nn.utils.rnn.pad_sequence(
            answer_ids, batch_first=True, padding_value=pad_token_id
        )

        # S(A|V): Visual Answer Scores
        with torch.inference_mode():
            outputs = model.generate(
                input_ids_with_images.to(args.device),
                images=torch.stack(image_tensors).half().to(args.device),
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=64,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
            )
            scores = torch.stack(outputs.scores, dim=0).permute(1, 0, 2).cpu()

        s_a_v = score_with_gaussian_weighting(
            scores, answer_ids, pad_token_id
        ).nan_to_num(nan=0.0)
        s_a_vs.append(s_a_v)

        # S(A): Direct Answer Scores (without image)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids_without_images.to(args.device),
                images=None,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=64,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
            )
            scores = torch.stack(outputs.scores, dim=0).permute(1, 0, 2).cpu()

        s_a = score_with_gaussian_weighting(
            scores, answer_ids, pad_token_id
        ).nan_to_num(nan=0.0)
        s_as.append(s_a)

    results = []
    for s_a_v, s_a in zip(s_a_vs, s_as):
        diff = torch.abs(s_a_v - s_a).sum(dim=1)
        results.append(diff)
    return torch.stack(results, dim=-1)


def get_last_index(csv_file_path):
    start_index = 0
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode="r") as csvfile:
            lines = csvfile.readlines()
        start_index = len(lines)
    return start_index


def I2C_gen(args):
    # Model setup
    tokenizer, model, image_processor, context_len = load_pretrained_model_from_path(
        args.model_path,
        args.model_base,
        args.load_4bit,
        args.use_flash_attn,
        torch_init=False,
    )

    # Dataset setup
    with open(os.path.expanduser(args.question_file), "r") as f:
        datasets = f.readlines()
    datasets = get_chunk(datasets, args.num_chunks, args.chunk_idx)

    # CSV file path
    csv_file_path = os.path.expanduser(args.i2c_file)

    # Determine the last valid processed index
    start_index = get_last_index(csv_file_path)

    with open(csv_file_path, "a") as csvfile:
        with tqdm(total=len(datasets), initial=start_index) as pbar:
            for i in range(start_index, len(datasets), args.batch_size):
                batch = [json.loads(line) for line in datasets[i : i + args.batch_size]]
                image_tensors = get_image_tensors(
                    args.dataset_path, image_processor, batch
                )

                # Compute I2C scores
                i2c_scores = compute_i2c(
                    args, model, tokenizer, image_tensors, batch, tokenizer.pad_token_id
                )

                with open(csv_file_path, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    for row in i2c_scores.tolist():  # row is a list of 5 elements
                        writer.writerow(row)  # write one row at a time
                pbar.update(args.batch_size)  # Update tqdm based on batch size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="/home/matthewwang16czap/models/llava-v1.5-7b"
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--question-file", type=str, default="./C3L/data/questions.jsonl"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="~/fiftyone/coco-2014/train/data"
    )
    parser.add_argument("--questions-num", type=int, default=5)
    parser.add_argument("--i2c-file", type=str, default="./C3L/data/I2C_test.csv")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--use-flash-attn", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    I2C_gen(args)
