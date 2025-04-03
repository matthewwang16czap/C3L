import argparse
import torch
import os
import json
from tqdm import tqdm
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
import re
import numpy as np
from pathlib import Path


def get_chunk(lst, n, k):
    return np.array_split(lst, n)[k]


def inference(
    questions, images, image_tensors, tokenizer, model, convs, max_new_tokens, device
):
    batch_size = len(questions)
    input_ids_list = []
    for i in range(batch_size):
        qs = questions[i]
        if images[i] is not None:
            if model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        convs[i].append_message(convs[i].roles[0], qs)
        convs[i].append_message(convs[i].roles[1], None)
        prompt = convs[i].get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(device)
        input_ids_list.append(input_ids)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True)

    stop_str = (
        convs[0].sep if convs[0].sep_style != SeparatorStyle.TWO else convs[0].sep2
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=torch.stack(image_tensors).half().to(device),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    outputs = [
        (
            output.strip()[: -len(stop_str)].strip()
            if output.endswith(stop_str)
            else output.strip()
        )
        for output in outputs
    ]
    return outputs


def validate_data(data):
    try:
        data = json.loads(data) if isinstance(data, str) else data
        if not isinstance(data, dict):
            raise ValueError("Input must be a JSON object.")

        for section in ["questions", "answers"]:
            if not isinstance(data.get(section, {}), dict) or len(data[section]) != 5:
                raise ValueError(f"'{section}' must contain exactly 5 key-value pairs.")
            for key, value in data[section].items():
                if not re.match(f"{section[:-1]}[1-5]$", key, re.IGNORECASE):
                    raise ValueError(f"Invalid key '{key}' in '{section}'.")
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or any(x in value.lower() for x in ["question", "answer"])
                ):
                    raise ValueError(
                        f"Invalid value in '{section}'. Values must be non-empty and not contain 'question' or 'answer'."
                    )
        return True, data
    except (json.JSONDecodeError, ValueError) as e:
        return False, str(e)


def get_last_index(question_file):
    """Get the last valid index from the saved file."""
    if not os.path.exists(question_file):
        return 0  # If file doesn't exist, start from index 0

    last_index = -1  # Default to -1, so first entry will be 0

    with open(question_file, "r") as qs_file:
        for line in qs_file:
            try:
                entry = json.loads(line.strip())  # Load last valid JSON line
                if "index" in entry:
                    last_index = max(last_index, entry["index"])  # Track max index
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    return last_index + 1  # Next index to continue from


def question_jsonl_gen(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_4bit=args.load_4bit,
        use_flash_attn=False,
        offload_folder="offload",
    )

    # Dataset
    with open(os.path.expanduser(args.dataset_file), "r") as f:
        datasets = f.readlines()
    datasets = get_chunk(datasets, args.num_chunks, args.chunk_idx)

    question_file = os.path.expanduser(args.question_file)
    os.makedirs(os.path.dirname(question_file), exist_ok=True)

    # Get last processed index
    start_index = get_last_index(question_file)

    with open(question_file, "a") as qs_file:
        with tqdm(total=len(datasets), initial=start_index) as pbar:
            for i in range(start_index, len(datasets), args.batch_size):
                batch = [json.loads(line) for line in datasets[i : i + args.batch_size]]
                images = []
                image_tensors = []
                convs = [conv_templates[args.conv_mode].copy() for _ in batch]

                for data_sample in batch:
                    image_path = os.path.join(
                        Path(args.dataset_path).expanduser(),
                        args.dataset_prefix + data_sample["image"],
                    )
                    try:
                        image = Image.open(image_path)
                        image_tensor = image_processor.preprocess(
                            image, return_tensors="pt"
                        )["pixel_values"][0].to(args.device)
                        images.append(image)
                        image_tensors.append(image_tensor)
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        continue

                # 2 stages' max token size
                max_new_tokens = [128, 512]

                # Get first stage inference output
                descriptions = inference(
                    [
                        "Generate descriptions based on the image in one to three complete sentences."
                    ]
                    * len(images),
                    images,
                    image_tensors,
                    tokenizer,
                    model,
                    convs,
                    max_new_tokens[0],
                    args.device,
                )

                for j, desc in enumerate(descriptions):
                    if desc[-1] != ".":
                        descriptions[j] = desc.rsplit(".", 1)[0] + "."
                    convs[j].messages[-1][-1] = descriptions[j]

                # Get second stage inference output
                outputs = inference(
                    [
                        """
                    Generate five in-depth reasoning questions and then answer them based on the image.
                    Your response should be a JSON in this format:
                    {"questions": {"question1": "your first question", "question2": "your second question", ...},
                    "answers": {"answer1": "your first answer", "answer2": "your second answer", ...}}
                    """
                    ]
                    * len(images),
                    [None] * len(images),
                    image_tensors,
                    tokenizer,
                    model,
                    convs,
                    max_new_tokens[1],
                    args.device,
                )

                for j, output in enumerate(outputs):

                    output = re.sub(r",\s*([\]}])", r"\1", output)
                    validity, data = validate_data(output)
                    # If questions are invalid, skip it
                    if not validity:
                        continue

                    conversations = []
                    for k in range(1, 6):
                        conversations.append(
                            {"from": "USER", "value": data["questions"][f"question{k}"]}
                        )
                        conversations.append(
                            {
                                "from": "ASSISTANT",
                                "value": data["answers"][f"answer{k}"],
                            }
                        )
                    qs_file.write(
                        json.dumps(
                            {
                                "id": batch[j]["id"],
                                "image": batch[j]["image"],
                                "index": i + j,
                                "description": descriptions[j],
                                "conversations": conversations,
                            }
                        )
                        + "\n"
                    )
                pbar.update(args.batch_size)  # Update tqdm based on batch size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument("--dataset-file", type=str, default="./C3L/dataset.jsonl")
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    question_jsonl_gen(args)
