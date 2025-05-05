from datasets import load_dataset

# Load the dataset
dataset = load_dataset("liuhaotian/llava-bench-in-the-wild")

# Save the dataset to a local directory
dataset.save_to_disk("./playground/data/eval/llava-bench-in-the-wild")
