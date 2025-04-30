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
from pathlib import Path
import csv
from C3L.scripts.utils import (
    get_chunk,
    tokenize_input,
)


def compute_answer_prob(scores, answer_ids):
    batch_size, scores_seq_len, vocab_size = scores.shape
    _, answer_seq_len = answer_ids.shape

    if answer_seq_len > scores_seq_len:
        pad_size = answer_seq_len - scores_seq_len
        scores = F.pad(scores, (0, 0, 0, pad_size), value=0)
    elif answer_seq_len < scores_seq_len:
        scores = scores[:, :answer_seq_len, :]

    # Safe gathering
    safe_answer_ids = answer_ids.clone()
    safe_answer_ids[safe_answer_ids == IGNORE_INDEX] = 0

    gathered_scores = scores.gather(2, safe_answer_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (batch, seq_len)

    # Now mask only where needed
    gathered_scores = gathered_scores.masked_fill(answer_ids == IGNORE_INDEX, -1e9)

    dist = F.softmax(gathered_scores, dim=-1)

    return dist


def compute_i2c(args, model, tokenizer, image_tensors, batch_data):
    batch_size = len(image_tensors)
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
                conv_templates[args.conv_mode].copy(),
            )
            for answer in answers
        ]

        # Use pad_sequence to pad the sequences to the same length
        input_ids_with_images = torch.nn.utils.rnn.pad_sequence(
            input_ids_with_images, batch_first=True, padding_value=IGNORE_INDEX
        )

        input_ids_without_images = torch.nn.utils.rnn.pad_sequence(
            input_ids_without_images, batch_first=True, padding_value=IGNORE_INDEX
        )

        answer_ids = torch.nn.utils.rnn.pad_sequence(
            answer_ids, batch_first=True, padding_value=IGNORE_INDEX
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

        s_a_v = compute_answer_prob(scores, answer_ids)
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

        s_a = compute_answer_prob(scores, answer_ids)
        s_as.append(s_a)

    kl_divs = []
    for s_a_v, s_a in zip(s_a_vs, s_as):
        min_len = min(s_a_v.shape[1], s_a.shape[1])
        s_a_v = s_a_v[:, :min_len]
        s_a = s_a[:, :min_len]
        kl = F.kl_div(
            F.log_softmax(s_a_v, dim=-1), s_a, reduction="none", log_target=False
        ).sum(dim=-1)
        kl_divs.append(kl)
    return torch.stack(kl_divs, dim=-1)


def I2C_gen(args):
    # Model setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_4bit=args.load_4bit,
        use_flash_attn=False,
        offload_folder="./offload",
    )

    # Dataset setup
    with open(os.path.expanduser(args.question_file), "r") as f:
        datasets = f.readlines()
    datasets = get_chunk(datasets, args.num_chunks, args.chunk_idx)

    # get image tensor shape
    image_tensor_shape = image_processor.preprocess(
        Image.open(
            os.path.join(
                Path(args.dataset_path).expanduser(),
                json.loads(datasets[0])["image"],
            )
        ),
        return_tensors="pt",
    )["pixel_values"][0].shape

    # CSV file path
    csv_file_path = os.path.expanduser(args.save_path)

    # Determine the last valid processed index
    start_index = 0
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode="r") as csvfile:
            lines = csvfile.readlines()
        start_index = len(lines)

    with open(csv_file_path, "a") as csvfile:
        with tqdm(total=len(datasets), initial=start_index) as pbar:
            for i in range(start_index, len(datasets), args.batch_size):
                batch = [json.loads(line) for line in datasets[i : i + args.batch_size]]

                image_tensors = []
                for data_sample in batch:
                    image_path = os.path.join(
                        Path(args.dataset_path).expanduser(),
                        data_sample["image"],
                    )
                    try:
                        image = Image.open(image_path)
                        image_tensor = image_processor.preprocess(
                            image, return_tensors="pt"
                        )["pixel_values"][0]
                        image_tensors.append(image_tensor)
                    except Exception as e:
                        print(f"Error processing image: {e}")

                # Compute I2C scores
                i2c_scores = compute_i2c(args, model, tokenizer, image_tensors, batch)

                with open(csv_file_path, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    for row in i2c_scores.tolist():  # row is a list of 5 elements
                        writer.writerow(row)  # write one row at a time
                pbar.update(args.batch_size)  # Update tqdm based on batch size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="/home/matthew/models/llava-v1.5-7b"
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--question-file", type=str, default="./C3L/data/questions.jsonl"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="~/fiftyone/coco-2014/train/data"
    )
    parser.add_argument("--questions-num", type=int, default=5)
    parser.add_argument("--save-path", type=str, default="./C3L/data/I2C.csv")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    I2C_gen(args)
