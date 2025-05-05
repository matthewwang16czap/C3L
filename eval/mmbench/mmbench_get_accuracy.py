import pandas as pd
import argparse


def mmbench_get_accuracy(args):
    # Step 1: Load the Excel file
    df = pd.read_excel(args.file_path)  # Replace with your actual filename

    # Step 2: Filter only valid predictions (A, B, C, D)
    valid_choices = {"A", "B", "C", "D"}
    df_valid = df[
        df["prediction"].isin(valid_choices) & df["answer"].isin(valid_choices)
    ]

    # Step 3: Compute accuracy
    accuracy = (df_valid["prediction"] == df_valid["answer"]).mean()

    print(f"Valid predictions: {len(df_valid)}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        type=str,
        default="./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/llava-v1.5-7b.xlsx",
    )
    args = parser.parse_args()

    mmbench_get_accuracy(args)
