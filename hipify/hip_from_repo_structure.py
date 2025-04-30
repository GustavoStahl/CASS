import os
import subprocess
import argparse
from tqdm import tqdm
from datasets import Dataset, config
from concurrent.futures import ThreadPoolExecutor

def process_file(record, source_folder, destination_folder):
    blob_id = record["blob_id"]
    extension = record["extension"]
    repo_name = record["repo_name"]
    branch_name = record["branch_name"]
    relative_path = record["path"].lstrip("/")  # Ensure no leading "/"

    # Construct paths
    source_file = os.path.join(source_folder, repo_name, os.path.basename(branch_name), relative_path)
    destination_dir = os.path.join(destination_folder, repo_name, os.path.basename(branch_name), os.path.dirname(relative_path))
    destination_file = os.path.join(destination_folder, repo_name, os.path.basename(branch_name), relative_path)
    destination_file = os.path.splitext(destination_file)[0] + ".hip"

    # Create necessary directories
    os.makedirs(destination_dir, exist_ok=True)

    command = f"./hipify-clang --clang-resource-directory /usr/lib/llvm-18/lib/clang/18 --cuda-gpu-arch sm_80 -o {destination_file} {source_file}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    except Exception:
        pass

def create_structure(dataset, source_folder, destination_folder, num_threads=8):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda record: process_file(record, source_folder, destination_folder), dataset), total=len(dataset), desc="Processing"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map StackV2 files to their original repository structure.")
    parser.add_argument("--source", required=True, type=str, help="Path to the source folder containing files")
    parser.add_argument("--destination", required=True, type=str, help="Path to the destination folder")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for parallel execution")

    args = parser.parse_args()

    dataset = Dataset.from_file(os.path.join(config.HF_DATASETS_CACHE, "bigcode___the-stack-v2-dedup/Cuda/0.0.0/94d47b4385264b30f228e28a5d63e9b2eee8c2c5/the-stack-v2-dedup-train.arrow"))

    create_structure(dataset, args.source, args.destination, args.threads)
