import argparse
import torch
import os
import json
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from PIL import Image
import re
import random
from pathlib import Path
from C3L.scripts.utils import (
    get_chunk,
    tokenize_input,
    instruction_prompts,
)


def inference(
    args,
    questions,
    have_images,
    image_tensors,
    tokenizer,
    model,
    convs,
    max_new_tokens,
):
    batch_size = len(questions)
    input_ids_list = []
    for i in range(batch_size):
        input_ids = tokenize_input(
            questions[i],
            have_images[i],
            tokenizer,
            model.config.mm_use_im_start_end,
            convs[i],
        )
        input_ids_list.append(input_ids)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True)

    stop_str = (
        convs[0].sep if convs[0].sep_style != SeparatorStyle.TWO else convs[0].sep2
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.to(args.device),
            images=torch.stack(image_tensors).half().to(args.device),
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
    for i, output in enumerate(outputs):
        convs[i].messages[-1][-1] = output
    return outputs


def process_qa_lists(responses):
    all_qa_lists = []
    for entry in responses:
        # Split on numbered entries like 1. 2. etc.
        qa_blocks = re.split(r"\n*\s*\d+\.\s*", entry.strip())
        qa_blocks = [b.strip() for b in qa_blocks if b.strip()]

        qa_list = []
        for block in qa_blocks:
            # Find the first "label: content" pair (we treat the first as the question and the second as the answer)
            parts = re.split(r"\s*\b\w+:\s*", block)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                question, answer = parts[0], parts[1]
                qa_list.append({"from": "human", "value": question})
                qa_list.append({"from": "gpt", "value": answer})
        all_qa_lists.append(qa_list)
    return all_qa_lists


def get_last_index(jsonl_file):
    """Get the last valid index from the saved file."""
    if not os.path.exists(jsonl_file):
        return 0  # If file doesn't exist, start from index 0

    last_index = -1  # Default to -1, so first entry will be 0

    with open(jsonl_file, "r") as qs_file:
        for line in qs_file:
            try:
                entry = json.loads(line.strip())  # Load last valid JSON line
                if "index" in entry:
                    last_index = max(last_index, entry["index"])  # Track max index
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    return last_index + 1  # Next index to continue from


def questions_gen(args):
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
        offload_folder="./offload",
    )

    # Dataset
    with open(os.path.expanduser(args.dataset_file), "r") as f:
        datasets = f.readlines()
    datasets = get_chunk(datasets, args.num_chunks, args.chunk_idx)

    question_file = os.path.expanduser(args.question_file)
    os.makedirs(os.path.dirname(question_file), exist_ok=True)

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

    # Get last processed index
    start_index = get_last_index(question_file)

    with open(question_file, "a") as qs_file:
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

                # 1 stages' max token size
                max_new_tokens = [512]

                # generate instructions
                instructions = [
                    random.choice(instruction_prompts) for _ in image_tensors
                ]

                # instruction question
                outputs = inference(
                    args,
                    instructions,
                    [True] * len(image_tensors),
                    image_tensors,
                    tokenizer,
                    model,
                    [conv_templates[args.conv_mode].copy() for _ in batch],
                    max_new_tokens[0],
                )
                qa_lists = process_qa_lists(outputs)

                for j, qa_list in enumerate(qa_lists):
                    if len(qa_list) == args.questions_num * 2:
                        qs_file.write(
                            json.dumps(
                                {
                                    "id": batch[j]["id"],
                                    "image": batch[j]["image"],
                                    "index": i + j,
                                    "instruction": instructions[j],
                                    "conversations": qa_list,
                                }
                            )
                            + "\n"
                        )
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
    parser.add_argument("--dataset-file", type=str, default="./C3L/data/dataset.jsonl")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/matthew/fiftyone/coco-2014/train/data",
    )
    parser.add_argument("--questions-num", type=int, default=5)
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
    questions_gen(args)
