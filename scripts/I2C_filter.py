import json
import math
import argparse
from utils import add_image_token


def extract_top_k_percent_conversations(args):
    all_items = []
    gpt_info = []  # (sample_idx, gpt_idx_in_conversations, score)

    # Step 1: Read all data
    with open(args.question_file, "r") as f_jsonl, open(args.i2c_file, "r") as f_csv:
        for sample_idx, (json_line, csv_line) in enumerate(zip(f_jsonl, f_csv)):
            item = json.loads(json_line)
            scores = list(map(float, csv_line.strip().split(",")))

            conversations = item["conversations"]
            gpt_indices = [
                i for i, conv in enumerate(conversations) if conv["from"] == "gpt"
            ]

            assert len(gpt_indices) == len(
                scores
            ), f"Mismatch scores and gpt turns at sample {sample_idx}"

            for gpt_idx, score in zip(gpt_indices, scores):
                gpt_info.append((sample_idx, gpt_idx, score))

            all_items.append(item)

    # Step 2: Sort globally
    gpt_info.sort(key=lambda x: x[2], reverse=True)

    # Step 3: Select top-k%
    total_gpt_turns = len(gpt_info)
    num_keep = max(1, math.ceil(total_gpt_turns * args.k))
    selected = set(
        (sample_idx, gpt_idx) for sample_idx, gpt_idx, _ in gpt_info[:num_keep]
    )

    # Step 4: Rebuild according to save_mode
    output_data = []

    for sample_idx, item in enumerate(all_items):
        conversations = item["conversations"]

        if args.save_mode == "full":
            new_conversations = []
            for i, conv in enumerate(conversations):
                if (sample_idx, i) in selected:
                    human_turn = conversations[i - 1]
                    human_turn["value"] = add_image_token(
                        human_turn["value"], args.mm_use_im_start_end
                    )
                    gpt_turn = conv
                    new_conversations.append(human_turn)
                    new_conversations.append(gpt_turn)
            item["conversations"] = new_conversations
            output_data.append(item)

            # Step 5: Write output
            with open(args.save_file, "w") as f_out:
                for item in output_data:
                    f_out.write(json.dumps(item) + "\n")

        elif args.save_mode == "turn":
            for i, conv in enumerate(conversations):
                if (sample_idx, i) in selected:
                    # Create a new item per turn
                    human_turn = conversations[i - 1]
                    human_turn["value"] = add_image_token(
                        human_turn["value"], args.mm_use_im_start_end
                    )
                    gpt_turn = conv

                    new_item = {
                        "id": item["id"],
                        "image": item["image"],
                        "conversations": [human_turn, gpt_turn],
                    }
                    output_data.append(new_item)

            # Step 5: Write output
            with open(args.save_file, "w") as f_out:
                f_out.write(json.dumps(output_data))

        else:
            raise ValueError(
                f"Invalid save_mode: {args.save_mode}. Must be 'full' or 'turn'."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mm-use-im-start-end", type=bool, default=False)
    parser.add_argument(
        "--question-file", type=str, default="./C3L/data/questions.jsonl"
    )
    parser.add_argument("--i2c-file", type=str, default="./C3L/data/I2C.csv")
    parser.add_argument(
        "--save-file", type=str, default="./C3L/data/filtered_questions.json"
    )
    parser.add_argument("--save-mode", type=str, default="turn")
    parser.add_argument("--k", type=float, default=0.1)
    args = parser.parse_args()

    extract_top_k_percent_conversations(args)
