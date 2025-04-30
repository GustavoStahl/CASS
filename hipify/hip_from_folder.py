import os
import subprocess
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def hipify_file(filename, source_folder, destination_folder):
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, os.path.splitext(filename)[0] + ".hip")
    
    os.makedirs(destination_folder, exist_ok=True)

    command = f"./hipify-clang --clang-resource-directory /usr/lib/llvm-18/lib/clang/18 --cuda-gpu-arch sm_80 -o {destination_file} {source_file}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    except Exception as e:
        pass
        # print(f"Failed to hipify {filename}: {e}")

def process_all(source_folder, destination_folder, num_threads=8):
    cuda_files = [f for f in os.listdir(source_folder) if f.endswith(".cu")]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(
            lambda f: hipify_file(f, source_folder, destination_folder), cuda_files),
            total=len(cuda_files),
            desc="Hipifying CUDA files"
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hipify flat directory of CUDA files.")
    parser.add_argument("--source", required=True, type=str, help="Path to the source folder")
    parser.add_argument("--destination", required=True, type=str, help="Path to the destination folder")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for parallel processing")

    args = parser.parse_args()
    process_all(args.source, args.destination, args.threads)