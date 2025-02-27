import argparse
import torch
import os
import json
from tqdm import tqdm


def question_reorder(args):
    # get dataset
    datasets = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        for line in f:
            datasets.append(line)

    # load I2C
    I2C_tensor = torch.load(args.i2c_file)

    sorted_I2C_tensor, sorted_indices = torch.sort(I2C_tensor, dim=1, descending=True)

    output_file = open(args.output_file, "w")
    # get reordered datasets
    for row, line in enumerate(tqdm(datasets)):
        data_sample = json.loads(line)
        output_file.write(
            json.dumps(
                {
                    "id": data_sample["id"],
                    "image": data_sample["image"],
                    "description": data_sample["description"],
                    "conversations": [
                        data_sample["conversations"][i] for i in sorted_indices[row]
                    ],
                }
            )
            + "\n"
        )
    output_file.flush()
    output_file.close()
    torch.save(sorted_I2C_tensor, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument(
        "--output-file", type=str, default="./C3L/reordered_question.jsonl"
    )
    parser.add_argument("--i2c-file", type=str, default="./C3L/I2C.pt")
    parser.add_argument("--save-path", type=str, default="./C3L/reordered_I2C.pt")
    args = parser.parse_args()

    question_reorder(args)
