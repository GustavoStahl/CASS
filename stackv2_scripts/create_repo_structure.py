import os
import shutil
import argparse
from tqdm import tqdm
from datasets import load_dataset

"""
This file maps all the codefiles from StackV2 CUDA saved in a folder with the
filename: <blob_id>.<extension> to an organized folder structure mimicking the 
repository they were taken from, as such:
<destination_dir>/<repo_name>/<path>,

e.g.:
dataset["repo_name"] = "MLauper/GPGPU"
dataset["path"] = "/CUDA/cuda_benchmark.cu"
result = "MLauper/GPGPU/CUDA/cuda_benchmark.cu"
"""

def create_structure(dataset, source_folder, destination_folder):

    existing_files_counter = 0
    for record in tqdm(dataset):
        blob_id = record["blob_id"]
        extension = record["extension"]
        repo_name = record["repo_name"]
        branch_name = record["branch_name"]
        relative_path = record["path"].lstrip("/")  # Ensure no leading "/"

        # Construct paths
        source_file = os.path.join(source_folder, f"{blob_id}.{extension}")
        destination_dir  = os.path.join(destination_folder, repo_name, os.path.basename(branch_name), os.path.dirname(relative_path))
        destination_file = os.path.join(destination_folder, repo_name, os.path.basename(branch_name), relative_path)

        # Create necessary directories
        os.makedirs(destination_dir, exist_ok=True)

        if not os.path.exists(destination_file): 
            shutil.copy2(source_file, destination_file)
        else:
            existing_files_counter += 1

    print(f"Files remap finished. Encoutered {existing_files_counter} files with the same name and path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map StackV2 files to their original repository structure.")
    parser.add_argument("--source", required=True, type=str, 
                        help="Path to the source folder containing files")
    parser.add_argument("--destination", required=True, type=str, 
                        help="Path to the destination folder")

    args = parser.parse_args()

    # Load dataset (assuming JSON format, modify as needed)
    dataset = load_dataset("bigcode/the-stack-v2-dedup", 
                           data_files="data/Cuda/train-00000-of-00001.parquet", 
                           split="train")

    create_structure(dataset, args.source, args.destination)