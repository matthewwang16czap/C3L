import argparse
import torch
import os
import json


def I2C_question_filter(args, ratio=0.1):
    # get dataset
    datasets = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        for line in f:
            datasets.append(line)

    # load I2C
    I2C_tensor = torch.load(args.i2c_file)

    # get indices with high i2c scores
    I2C_tensor_flattened = I2C_tensor.flatten()
    k = max(1, int(ratio * I2C_tensor_flattened.numel()))
    topk_values, topk_indices = torch.topk(I2C_tensor_flattened, k)

    rows = topk_indices // I2C_tensor.shape[1]
    cols = topk_indices % I2C_tensor.shape[1]
    sorted_indices = torch.argsort(rows)
    rows = rows[sorted_indices]
    cols = cols[sorted_indices]

    output_data = []
    cur_row = rows[0].item()
    data_sample = json.loads(datasets[cur_row])
    filtered_qs_nums = []

    for idx in range(len(rows)):
        row = rows[idx].item()
        col = cols[idx].item()

        if row > cur_row:
            output_data.append(
                {
                    "id": data_sample["id"],
                    "image": args.dataset_prefix + data_sample["image"],
                    "conversations": [
                        {
                            "from": (
                                "human"
                                if data_sample["conversations"][i]["from"] == "USER"
                                else "gpt"
                            ),
                            "value": data_sample["conversations"][i]["value"],
                        }
                        for i in filtered_qs_nums
                    ],
                }
            )
            cur_row = row
            data_sample = json.loads(datasets[cur_row])
            filtered_qs_nums = []

        filtered_qs_nums.append(2 * col)
        filtered_qs_nums.append(2 * col + 1)

    # add last data sample
    output_data.append(
        {
            "id": data_sample["id"],
            "image": args.dataset_prefix + data_sample["image"],
            "conversations": [
                {
                    "from": (
                        "human"
                        if data_sample["conversations"][i]["from"] == "USER"
                        else "gpt"
                    ),
                    "value": data_sample["conversations"][i]["value"],
                }
                for i in filtered_qs_nums
            ],
        }
    )

    # Write as a single JSON list
    with open(args.output_file, "w") as out_f:
        json.dump(output_data, out_f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument(
        "--output-file", type=str, default="./C3L/high_I2C_questions.json"
    )
    parser.add_argument("--dataset-prefix", type=str, default="COCO_train2014_")
    parser.add_argument("--i2c-file", type=str, default="./C3L/I2C.pt")
    args = parser.parse_args()

    I2C_question_filter(args)
