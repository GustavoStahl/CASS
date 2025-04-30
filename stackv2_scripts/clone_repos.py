import os
import shutil
import argparse
import subprocess
from tqdm import tqdm
from datasets import Dataset, config
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

def clone_repo(repo_name, branch_name, destination):
    url = f"https://github.com/{repo_name}.git"
    destination_dir = destination

    os.makedirs(destination_dir, exist_ok=True)
    
    destination_dir = os.path.join(destination, repo_name, branch_name)

    repo_exists = os.path.exists(destination_dir)
    if(repo_exists):
        shutil.move(destination_dir, destination_dir + "_tmp")
    try:
        subprocess.run(f"git clone -b {branch_name} {url} {destination_dir}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        if(repo_exists):
            shutil.rmtree(destination_dir + "_tmp")
    except subprocess.CalledProcessError as e:
        print(e.returncode, e.stderr.decode())
        if(repo_exists):
            shutil.rmtree(destination_dir)
            shutil.move(destination_dir + "_tmp", destination_dir)
    except Exception:
        if(repo_exists):
            shutil.rmtree(destination_dir)
            shutil.move(destination_dir + "_tmp", destination_dir)

def main(destination, first_n):
    arrow_path = "bigcode___the-stack-v2-dedup/Cuda/0.0.0/94d47b4385264b30f228e28a5d63e9b2eee8c2c5/the-stack-v2-dedup-train.arrow"
    dataset = Dataset.from_file(os.path.join(config.HF_DATASETS_CACHE, arrow_path))
    analysis = {}
    for data in tqdm(dataset):
        repo_name = data["repo_name"]
        branch_name = data["branch_name"]
        branch_name = os.path.basename(branch_name)

        if repo_name in analysis:
            analysis[repo_name]["file_count"] += 1
            if branch_name not in analysis[repo_name]["branch_names"]:
                analysis[repo_name]["branch_names"].append(branch_name)
        else:
            analysis[repo_name] = {"file_count": 1, "branch_names": [branch_name]}

    analysis_sorted = OrderedDict(sorted(analysis.items(), key=lambda item: item[1]["file_count"], reverse=True))
    first_n_repos = list(analysis_sorted.items())[:first_n]

    #NOTE: synchronous execution 
    # url_format = "https://github.com/{}.git"
    # for repo_name, count_and_branch_names in first_n_repos:
    #     url = url_format.format(repo_name)

    #     branch_names = count_and_branch_names["branch_names"]
    #     for branch_name in branch_names:
    #         destination_dir = os.path.join(destination, repo_name, branch_name)
    #         subprocess.run(f"git clone -b {branch_name} {url} {destination_dir}", shell=True)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(clone_repo, repo_name, branch_name, destination): (repo_name, branch_name)
                   for repo_name, count_and_branch_names in first_n_repos
                   for branch_name in count_and_branch_names["branch_names"]}
        
        with tqdm(total=len(futures), desc="Cloning repositories") as pbar:
            for future in as_completed(futures):
                repo_name, branch_name = futures[future]
                future.result()
                pbar.update(1)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone the first N repositories with the highes file count.")
    parser.add_argument("--destination", required=True, type=str, 
                        help="Path to the destination folder")
    parser.add_argument("--first-n", "-N", default=100, type=int, 
                        help="How many repositories to clone")    

    args = parser.parse_args()

    main(args.destination, args.first_n)