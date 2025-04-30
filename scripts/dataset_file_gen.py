import os
import re
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def process_file(filename, prefix):
    delimiters = [prefix, ".", ":"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    _, id, filetype, *rest = re.split(regex_pattern, filename)
    return {"id": id, "image": filename}


def dataset_file_gen(
    dir=Path("~/fiftyone/coco-2014/train/data").expanduser(),
    prefix="COCO_train2014_",
    out_dir="./C3L/data/dataset.jsonl",
):
    # Get all filenames in the directory
    filenames = os.listdir(dir)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Map filenames to the processing function in parallel
        results = list(executor.map(process_file, filenames, [prefix] * len(filenames)))

    # Write results to a JSONL file
    with open(out_dir, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    dataset_file_gen()
