import os
import subprocess
import sys

DATASET_PATH = "C3L/data/dataset.jsonl"


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
    # Step 1: Check dataset
    if not os.path.exists(DATASET_PATH):
        print("📂 dataset.jsonl not found. Generating dataset...")
        run_python_script("dataset_file_gen.py")
    else:
        print("✅ dataset.jsonl already exists. Skipping dataset generation.")

    # Step 2: Generate questions
    run_python_script("questions_gen.py")

    # Step 3: Generate I2C pairs
    run_python_script("I2C_gen.py")

    # Step 4: Run contrastive learning
    run_python_script("contrastive_learn.py")

    # Step 5: Fine-tune SDPA model
    run_shell_script("finetune_sdpa.sh")

    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
