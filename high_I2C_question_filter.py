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

    # Get the number of top elements (10% of total)
    k = max(1, int(ratio * I2C_tensor_flattened.numel()))  # Ensure at least 1 element

    # Get the top-k indices in the flattened tensor
    topk_values, topk_indices = torch.topk(I2C_tensor_flattened, k)

    # Convert flattened indices to 2D indices
    rows = topk_indices // I2C_tensor.shape[1]  # Row indices
    cols = topk_indices % I2C_tensor.shape[1]  # Column indices

    # Sort based on row values
    sorted_indices = torch.argsort(rows)
    rows = rows[sorted_indices]
    cols = cols[sorted_indices]

    output_file = open(args.output_file, "w")
    # get filtered datasets
    cur_row = rows[0]
    data_sample = json.loads(datasets[cur_row])
    filtered_qs_nums = []
    for idx in range(len(rows)):
        row = rows[idx]
        col = cols[idx]
        if row > cur_row:
            # write filtered data to output
            output_file.write(
                json.dumps(
                    {
                        "id": data_sample["id"],
                        "image": data_sample["image"],
                        "conversations": [
                            data_sample["conversations"][i] for i in filtered_qs_nums
                        ],
                    }
                )
                + "\n"
            )
            output_file.flush()
            # refresh
            cur_row = row
            data_sample = json.loads(datasets[cur_row])
            filtered_qs_nums = []
        filtered_qs_nums.append(2 * col)
        filtered_qs_nums.append(2 * col + 1)
    # add last data sample
    output_file.write(
        json.dumps(
            {
                "id": data_sample["id"],
                "image": data_sample["image"],
                "conversations": [
                    data_sample["conversations"][i] for i in filtered_qs_nums
                ],
            }
        )
        + "\n"
    )
    output_file.flush()
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument(
        "--output-file", type=str, default="./C3L/high_I2C_question.jsonl"
    )
    parser.add_argument("--i2c-file", type=str, default="./C3L/I2C.pt")
    args = parser.parse_args()

    I2C_question_filter(args)
