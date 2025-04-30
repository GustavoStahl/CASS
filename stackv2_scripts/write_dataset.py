# import json
# import os
# import boto3
# import multiprocessing
# from smart_open import open
# from datasets import load_dataset
# from botocore import UNSIGNED
# from botocore.client import Config
# from tqdm import tqdm

# to_download = ["Cuda"]

# done_blobs = {}
# def collect_downloaded_blob_ids(lang_subdir):
#     global done_blobs
#     done_blobs = {}
#     if not os.path.exists(lang_subdir):
#         return
#     for filename in os.listdir(lang_subdir):
#         try:
#             if filename.startswith('done_') and filename.endswith('.json'):
#                 filepath = os.path.join(lang_subdir, filename)
#                 with open(filepath, 'r') as file:
#                     data = json.load(file)
#                     for blob_id in data['blob_id']:
#                         done_blobs[blob_id] = 1
#         except:
#             continue
#     print(f"Already downloaded blobs: {len(done_blobs)}")

# def download_chunk(data_repo, download_folder, worker_id, num_workers):
#     global done_blobs
#     cur_done_blobs = []     #helpful in resuming the interrupted runs
#     s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
#     print(f"Starting {data_repo} download for {worker_id}")
#     ds = load_dataset(data_repo, split="train", streaming=True)
#     print(f"Filtering ds for {worker_id}")
#     ds = ds.filter(lambda row, idx: idx % num_workers == worker_id, with_indices=True)

#     data_jsonl = []
#     for i, row in tqdm(enumerate(ds), desc=f"Worker {worker_id}"):
#         blob_id, src_encoding, language = row["blob_id"], row["src_encoding"], row['language']
#         if blob_id in done_blobs:
#             #print(f"{blob_id} already downloaded")
#             continue
#         s3_url = f"s3://softwareheritage/content/{blob_id}"
#         try:
#             with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
#                 content = fin.read().decode(src_encoding)
#         except Exception as e:
#             print(f"Exception occured: {e}")
#             continue
#         data_jsonl.append({"text": content})
#         cur_done_blobs.append(blob_id)

#         #store 8K records in each jsonl file
#         if len(data_jsonl) ==  8000:
#             directory = os.path.join(download_folder, language)
#             if not os.path.exists(directory):
#                 os.makedirs(directory)
#             data_path = os.path.join(directory, blob_id + ".jsonl") #save with current blob_id for uniqueness
#             write_dicts_to_jsonl(data_jsonl, data_path)
#             data_jsonl = []

#             #write blob_ids for blobs done being downloaded and written
#             with open(data_path.replace(blob_id + ".jsonl", f"done_{blob_id}.json"), "w") as f:
#                 json.dump({"blob_id": cur_done_blobs}, f)
#             cur_done_blobs = []

#     # Save any remaining data
#     if data_jsonl:
#         directory = os.path.join(download_folder, language)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         data_path = os.path.join(directory, f"remaining_{worker_id}.jsonl")
#         write_dicts_to_jsonl(data_jsonl, data_path)

#         #write blob_ids for blobs done being downloaded and written
#         with open(os.path.join(directory, f"done_{blob_id}.json"), "w") as f:
#             json.dump({"blob_id": cur_done_blobs}, f)

# def download_the_stack_v2(data_repo, download_folder, num_workers):
#     for lang in to_download:
#         lang_out_subdir = os.path.join(download_folder, lang)
#         lang_subdir = os.path.join(data_repo, lang)
#         collect_downloaded_blob_ids(lang_out_subdir)
#         with multiprocessing.Pool(processes=num_workers) as pool:
#             pool.starmap(download_chunk, [(lang_subdir, download_folder, i, num_workers) for i in range(num_workers)])

# def write_dicts_to_jsonl(dict_list, jsonl_path):
#     print("Writing ", jsonl_path)
#     with open(jsonl_path, "w") as f:
#         for d in dict_list:
#             json.dump(d, f)
#             f.write("\n")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="the-stack-v2 download entry.")
#     parser.add_argument("--hf_data_dir", type=str)
#     parser.add_argument("--download_folder", type=str)
#     parser.add_argument("--num_workers", type=int)
#     args = parser.parse_args()
#     download_the_stack_v2(args.hf_data_dir, args.download_folder, args.num_workers)

import os
import boto3
import threading
from botocore import UNSIGNED
from botocore.client import Config
from smart_open import open
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# S3 client with unsigned access
s3 = boto3.client('s3', region_name='me-central-1', config=Config(signature_version=UNSIGNED))

def download_and_save(r, save_dir):
    """
    Downloads content from S3 and saves it to a local directory.
    
    Args:
        r (dict): Row containing 'blob_id', 'src_encoding', and 'extension'.
        save_dir (str): Directory where the file should be saved.
    """
    blob_id, src_encoding, extension = r["blob_id"], r["src_encoding"], r["extension"]
    s3_url = f"s3://softwareheritage/content/{blob_id}"

    try:
        # Download content
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as s3bucket:
            content = s3bucket.read().decode(src_encoding)

        # Define save path
        save_path = os.path.join(save_dir, f"{blob_id}.{extension}")

        # Save content to file
        with open(save_path, "w", encoding=src_encoding) as f:
            f.write(content)

        return True  # Successful download

    except Exception as e:
        print(f"Failed to download {blob_id}: {e}")
        return False  # Failed download

def main(save_dir, num_threads=8):
    """
    Main function to download multiple files in parallel.
    
    Args:
        save_dir (str): Directory where files will be saved.
        num_threads (int): Number of threads for parallel downloads.
    """
    dataset = load_dataset("bigcode/the-stack-v2-dedup", 
                           data_files="data/Cuda/train-00000-of-00001.parquet", 
                           split="train")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        progress = tqdm(desc="Starting downloads", total=len(dataset), unit="file", dynamic_ncols=True)

        for r in dataset:
            # Submit each file download to the thread pool
            future = executor.submit(download_and_save, r, save_dir)
            futures.append(future)

            # Update progress dynamically
            progress.update(1)
        progress.close()

        # Ensure all downloads are completed
        progress = tqdm(desc="Waiting downloads", total=len(futures), unit="file", dynamic_ncols=True)
        for future in as_completed(futures):
            future.result()  # Raise any exception inside threads
            progress.update(1)
        progress.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-threaded S3 file downloader")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save downloaded files")
    parser.add_argument("--num-threads", type=int, default=8, help="Number of threads to use")

    args = parser.parse_args()
    main(args.save_dir, args.num_threads)

# import os
# import boto3
# from smart_open import open
# from datasets import load_dataset

# def download_contents(s3, blob_id, src_encoding):
#     s3_url = f"s3://softwareheritage/content/{blob_id}"
    
#     with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
#         content = fin.read().decode(src_encoding)
    
#     return {"content": content}

# def main():
#     session = boto3.Session(
#         aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
#         aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
#     s3 = session.client("s3")

#     dataset_name = "bigcode/the-stack-v2-dedup"
#     dataset = load_dataset(dataset_name, split="train", data_files="data/Cuda/train-00000-of-00001.parquet")
#     dataset = dataset.map(lambda row: download_contents(s3, row["blob_id"], row["src_encoding"]))

#     for row in dataset:
#         print(row["content"])
#         break

#     import pdb; pdb.set_trace()

# if __name__ == "__main__":
#     main()