<div align="center">
  <img src="assets/cass-logo.png" width="11%" align="left"/>
</div>

<div style="margin-top:50px; margin-left: 12%;">
  <h1 style="font-size: 30px; margin: 0;"> CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark</h1>
</div>

This is the official repository for the CASS: **C**UDA-**A**MD A**ss**embly paper. Since the methodology used to construct our dataset consists of multiple independent modules, we have organized the scripts into separate folders to ensure clarity and modularity. For questions regarding individual components (e.g., Synthetic Pipeline, OpenCL Pipeline), please refer to their respective directories for tailored instructions.

## Hardware setup
Our scripts require a GPU to be present in the user’s machine. The CUDA portion of our pipeline requires an NVIDIA GPU, while the HIP portion requires an AMD GPU. The provided code was tested on an AMD RX 7900, an NVIDIA A100, and an NVIDIA RTX 4090.

## Execution environment 
To ensure reproducibility, we provide a Docker container that manages all dependencies. To build and run it, execute:
```bash
docker build -t transpiler .
docker compose run transpiler
```

## Overview
Here's a breakdown of our folder structure:
* stackv2_scripts: contains code for processing the Stack v2 dataset. Specifically, it fetches all available CUDA files, organizes them according to their original repository file-tree structure, clones repositories with the highest number of CUDA files, and extracts corresponding CPU-GPU assembly
* hipify: includes code for converting CUDA files to HIP using AMD’s HIPify tool.

## Dataset
Our finished dataset processed using the tools available in this repository can be accessed through the following links:
- [Dataset](https://huggingface.co/datasets/MBZUAI/cass)
- [Benchmark](https://huggingface.co/datasets/Sarim-Hash/cass_bench_new_one)
