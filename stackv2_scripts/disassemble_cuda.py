import os
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def compile_and_extract_sass(file_path, blob_id, sass_dir, arch):
    """Compiles a CUDA file and extracts SASS, saving it to a specified directory."""
    binary_path = os.path.join(sass_dir, blob_id + ".out") # Temporary binary file
    host_assembly_path = os.path.join(sass_dir, blob_id + ".s")
    device_assembly_path = os.path.join(sass_dir, blob_id + ".sass")

    default_nvcc_args = f"nvcc -std=c++17 -Xcompiler=-Os -DNDEBUG -w -arch={arch}"

    # compile_command = f"nvcc -std=c++17 -O1 -DNDEBUG -w -arch={arch} -cubin -o /dev/null \"{file_path}\""

    # host compilation commands
    host_assembly_command = f"{default_nvcc_args} -Xcompiler -S -o {host_assembly_path} -c {file_path}"
    clean_host_assembly_command = f"sed -i '/\\.nv_fatbin/,/\\.text/d' {host_assembly_path}"

    # device compilation commands
    device_binary_command = f"{default_nvcc_args} -cubin -o {binary_path} {file_path}"
    device_assembly_command = f"cuobjdump --dump-sass {binary_path} > {device_assembly_path}"

    compiled_file = None
    files_to_remove = [binary_path]
    try:
        # Compile CUDA file
        subprocess.run(host_assembly_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        subprocess.run(clean_host_assembly_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        subprocess.run(device_binary_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        subprocess.run(device_assembly_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)

        compiled_file = file_path

    except Exception:
        files_to_remove.extend([device_assembly_path, host_assembly_path])

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
    
    return compiled_file

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Compile CUDA files and extract SASS.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the CUDA dataset directory")
    parser.add_argument("--sass-dir", type=str, required=True, help="Directory to save SASS files")
    parser.add_argument("--arch", type=str, required=True, help="CUDA architecture (e.g., sm_80, sm_70, sm_86)")

    args = parser.parse_args()

    dataset_dir = args.dataset
    sass_dir = args.sass_dir
    arch = args.arch

    os.makedirs(sass_dir, exist_ok=True)  # Ensure SASS output directory exists

    skip_files = ["2477cd87d5ffce0cca3f3cdb7ced4f5726938a01", "51171989a13708efd400d9c7c9b9ee6cfcb12e11"]

    cuda_files = []
    from datasets import Dataset, config
    dataset = Dataset.from_file(os.path.join(config.HF_DATASETS_CACHE, "bigcode___the-stack-v2-dedup/Cuda/0.0.0/94d47b4385264b30f228e28a5d63e9b2eee8c2c5/the-stack-v2-dedup-train.arrow"))
    cuda_files = [[os.path.join(dataset_dir, d["repo_name"], os.path.basename(d["branch_name"]), d["path"].lstrip("/")), d["blob_id"]] 
                  for d in dataset
                  if not any([skip_file in d["blob_id"] for skip_file in skip_files]) and 
                  d["extension"] != "cuh"]

    if not cuda_files:
        print("No CUDA files found in the dataset directory.")
        return
    
    # import random
    # random.seed(123)
    # random.shuffle(cuda_files)
    # cuda_files = cuda_files[:30]    
    
    compiled_files = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(compile_and_extract_sass, file, blob_id, sass_dir, arch): file for (file, blob_id) in cuda_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CUDA files"):
            result = future.result(timeout=30)
            if result:
                compiled_files.append(result)  

    print(f"\nCompilation check completed! {len(compiled_files)} out of {len(cuda_files)} files compiled successfully.")
    print(f"SASS files saved in: {sass_dir}")

if __name__ == "__main__":
    main()
