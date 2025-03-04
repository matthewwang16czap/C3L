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
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import json
import re
import numpy as np
from pathlib import Path


def get_chunk(lst, n, k):
    return np.array_split(lst, n)[k]


# need to edit
def inference(qs, image, image_tensor, tokenizer, model, conv, max_new_token):
    if image is not None:
        # first message
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
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_new_token,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)].strip()

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


def question_jsonl_gen(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, load_4bit=args.load_4bit
    )
    # Dataset
    with open(os.path.expanduser(args.dataset_file), "r") as f:
        datasets = f.readlines()
    datasets = get_chunk(datasets, args.num_chunks, args.chunk_idx)
    question_file = os.path.expanduser(args.question_file)
    os.makedirs(os.path.dirname(question_file), exist_ok=True)
    with open(args.question_file, "w") as qs_file:
        qs_file = open(question_file, "w")
        line_index = 0
        for line in tqdm(datasets):
            line = json.loads(line)
            line_index = line_index + 1
            image = os.path.join(
                Path(args.dataset_path).expanduser(),
                args.dataset_prefix + line["image"],
            )
            image = Image.open(image)
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]

            # 2 stages' max token size
            max_new_tokens = [128, 512]

            multi_choice_data = {}
            multi_choice_data["id"] = line["id"]
            multi_choice_data["image"] = line["image"]
            multi_choice_data["conversations"] = []

            conv = conv_templates[args.conv_mode].copy()

            # get first stage inference output
            description = inference(
                "Generate descriptions based on the image in one to three complete sentence.",
                image,
                image_tensor,
                tokenizer,
                model,
                conv,
                max_new_tokens[0],
            )

            # remove imcomplete sentence
            if description[-1] != ".":
                description = description.split(description.split(".")[-1])[0]

            conv.messages[-1][-1] = description

            # get second stage inference output
            outputs = inference(
                """
                Generate five in-depth reasoning questions and then answer them based on the image.
                Your response should be a JSON in this JSON format:
                {
                    "questions": {
                        "question1": "your first question",
                        "question2": "your second question",
                        ...
                    },
                    "answers": {
                        "answer1": "your first answer",
                        "answer2": "your second answer",
                        ...
                    },
                }
                """,
                None,
                image_tensor,
                tokenizer,
                model,
                conv,
                max_new_tokens[1],
            )

            outputs = re.sub(r",\s*([\]}])", r"\1", outputs)

            conv.messages[-1][-1] = outputs
            validity, data = validate_data(outputs)

            # if questions is invalid, skip it
            if not validity:
                continue

            for i in range(1, 6):
                # Add human question
                human_dict = {
                    "from": "USER",
                    "value": data["questions"][f"question{i}"],
                }
                multi_choice_data["conversations"].append(human_dict)

                # Add llava answer
                llava_dict = {
                    "from": "ASSISTANT",
                    "value": data["answers"][f"answer{i}"],
                }
                multi_choice_data["conversations"].append(llava_dict)

            qs_file.write(
                json.dumps(
                    {
                        "id": line["id"],
                        "image": line["image"],
                        "description": description,
                        "conversations": multi_choice_data["conversations"],
                    }
                )
                + "\n"
            )


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
    parser.add_argument("--num-chunks", type=int, default=1000)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", type=bool, default=True)
    args = parser.parse_args()

    question_jsonl_gen(args)
