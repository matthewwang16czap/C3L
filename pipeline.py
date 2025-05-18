import os
import argparse
import subprocess
import sys

# Default paths (will be overridden by command-line arguments)
DEFAULT_MODEL_PATH = "/home/matthewwang16czap/models/llava-v1.5-7b"
DEFAULT_DATASET_DIR = "/home/matthewwang16czap/fiftyone/coco-2014/train/data"


def parse_arguments():
    """Parse command-line arguments for the entire pipeline"""
    parser = argparse.ArgumentParser(description="C3L Pipeline")

    # Pipeline control arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="C3L/data/dataset.jsonl",
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="Force regenerate dataset even if exists",
    )

    # Questions generation arguments
    questions_group = parser.add_argument_group("Question Generation")
    questions_group.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to LLaVA model"
    )
    questions_group.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Path to image dataset directory",
    )
    questions_group.add_argument(
        "--dataset-size", type=int, default=6000, help="Number of images to process"
    )
    questions_group.add_argument(
        "--questions-num", type=int, default=5, help="Questions per image"
    )
    questions_group.add_argument(
        "--batch-size", type=int, default=4, help="Inference batch size"
    )
    questions_group.add_argument(
        "--temperature", type=float, default=0.2, help="Generation temperature"
    )
    questions_group.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    return parser.parse_args()


def run_questions_gen(args):
    """Run question generation with parsed arguments"""
    script_name = "questions_gen.py"
    print(f"\n🔧 Running {script_name}...")

    cmd = [
        sys.executable,
        f"C3L/scripts/{script_name}",
        "--model-path",
        args.model_path,
        "--dataset-file",
        args.dataset_path,
        "--question-file",
        "C3L/data/questions.jsonl",
        "--dataset-path",
        args.dataset_dir,
        "--dataset-size",
        str(args.dataset_size),
        "--questions-num",
        str(args.questions_num),
        "--conv-mode",
        "llava_v1",
        "--num-chunks",
        "1",
        "--chunk-idx",
        "0",
        "--temperature",
        str(args.temperature),
        "--num_beams",
        "1",
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)


def run_python_script(script_name):
    print(f"\n🔧 Running {script_name}...")
    result = subprocess.run(
        [sys.executable, f"C3L/scripts/{script_name}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)


def run_shell_script(script_name):
    print(f"\n🔧 Running {script_name}...")
    result = subprocess.run(
        ["bash", f"C3L/scripts/{script_name}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)


def main():
    args = parse_arguments()

    # Step 1: Check dataset
    if not os.path.exists(args.dataset_path) or args.force_dataset:
        print("📂 Generating dataset...")
        run_python_script("dataset_file_gen.py")
    else:
        print("✅ dataset.jsonl already exists. Skipping generation.")

    # Step 2: Generate questions with configured arguments
    run_questions_gen(args)

    # Remaining steps (unchanged)
    run_python_script("I2C_gen.py")
    run_python_script("contrastive_learn.py")
    run_shell_script("finetune_sdpa.sh")

    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
