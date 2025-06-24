<div align="center">
  <img src="assets/cass-logo.png" width="11%" align="left"/>
</div>

<div style="margin-top:50px; margin-left: 12%;">
  <h1 style="font-size: 30px; margin: 0;"> CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark</h1>
</div>


<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" height="4"/>
</div>

<p align="center">
  <a href="https://gustavostahl.github.io/CASS/"><img src="https://img.shields.io/badge/Project-Website-87CEEB?style=flat-square" alt="Website"></a>
  <a href="https://arxiv.org/abs/2505.16968"><img src="https://img.shields.io/badge/arXiv-Paper-brightgreen?style=flat-square" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/MBZUAI/cass"><img src="https://img.shields.io/badge/🤗_Dataset-Access-green" alt="dataset"></a>
  <a href="https://huggingface.co/collections/ahmedheakl/cass-683efb1596a1dd802de1593a"><img src="https://img.shields.io/badge/HuggingFace-Model-F9D371" alt="model"></a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/ahmed-heakl/"><b>Ahmed Heakl</b></a>, 
  <a href="https://www.linkedin.com/in/gustavostahl/"><b>Gustavo Bertolo Stahl</b></a>,
  <a href="https://www.linkedin.com/in/sarim-hashmi-b10b35136/"><b>Sarim Hashmi</b></a>, 
  <a href="https://www.linkedin.com/in/eddieseunghunhan/"><b>Seung Hun Eddie Han</b></a><br> 
  <a href="https://salman-h-khan.github.io/"><b>Salman Khan</b></a>,
   <a href="https://ma3mool.github.io/"><b>Abdulrahman Mahmoud</b></a>,
</p>


<p align="center">
  <b>MBZUAI</b> · <b>Australia National University</b>
</p>

---

## 🆕 Latest Updates

- 📢 **June 2025**: Paper and inference code released!


## Overview

We introduce CASS, the first large-scale dataset and model suite for cross-architecture GPU code transpilation, targeting both source-level (CUDA ↔ HIP) and assembly-level (Nvidia SASS ↔ AMD RDNA3) translation. The dataset comprises 70k verified code pairs across host and device, addressing a critical gap in low-level GPU code portability. Leveraging this resource, we train the CASS family of domain-specific language models, achieving 95% source translation accuracy and 37.5% assembly translation accuracy, substantially outperforming commercial baselines such as GPT-4o, Claude, and Hipify. Our generated code matches native performance in over 85% of test cases, preserving runtime and memory behavior. To support rigorous evaluation, we introduce CASS-Bench, a curated benchmark spanning 16 GPU domains with ground-truth execution. All data, models, and evaluation tools are released as open source to foster progress in GPU compiler tooling, binary compatibility, and LLM-guided hardware translation.


<p align="center">
  <img src="assets/cass-main.png" width="100%" alt="CASS Overview" />
</p>

---


## (1) Data Construction Pipeline
Since the methodology used to construct our dataset consists of multiple independent modules, we have organized the scripts into separate folders to ensure clarity and modularity. For questions regarding individual components (e.g., Synthetic Pipeline, OpenCL Pipeline), please refer to their respective directories for tailored instructions.

### Compiler Stack

<p align="center">
  <img src="assets/gpu-compiler-stack.png" width="90%" alt="CASS Overview" />
</p>


> The Nvidia (left) and AMD (right) stacks illustrate the compilation process for CUDA and HIP. Blue denotes device-side steps; green denotes host-side steps. Nvidia’s stack is opaque; accessing device assembly (SASS) requires first compiling to binary, then using cuobjdump. In contrast, AMD’s process is transparent, allowing direct inspection and modification of device assembly (RDNA3) before host integration.

### Hardware setup
Our scripts require a GPU to be present in the user’s machine. The CUDA portion of our pipeline requires an NVIDIA GPU, while the HIP portion requires an AMD GPU. The provided code was tested on an AMD RX 7900, an NVIDIA A100, and an NVIDIA RTX 4090.

### Execution environment 
To ensure reproducibility, we provide a Docker container that manages all dependencies. To build and run it, execute:
```bash
docker build -t transpiler .
docker compose run transpiler
```

### Subfolders
Here's a breakdown of our folder structure:
* `stackv2_scripts`: contains code for processing the Stack v2 dataset. Specifically, it fetches all available CUDA files, organizes them according to their original repository file-tree structure, clones repositories with the highest number of CUDA files, and extracts corresponding CPU-GPU assembly
* `hipify`: includes code for converting CUDA files to HIP using AMD’s HIPify tool.

### Generated Data
Refer to [Huggingface Dataset and Benchmark](https://huggingface.co/datasets/MBZUAI/cass) for details on how to load the dataset and benchmark.


## (2) Inference
We provide a simple inference script to run the CASS models. The script supports both source-to-source and assembly-to-assembly translation. 

### Available Models
We provide multiple models with multiple parameter scales.
<table border="1" cellpadding="6" cellspacing="0">
  <tr>
    <th>Model</th>
    <th>Type</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>CASS-src-1.5B</td>
    <td rowspan="3">Source-to-Source</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-src-1.5b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-src-3B</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-src-3b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-src-7B</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-src-7b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-smA100-1.5B</td>
    <td rowspan="3">Assembly-to-Assembly (sm_80, A100)</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-smA100-1.5b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-smA100-3B</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-smA100-3b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-smA100-7B</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-smA100-7b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-sm4090-1.5B</td>
    <td rowspan="3">Assembly-to-Assembly (sm_89, RTX4090)</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-sm4090-1.5b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-sm4090-3B</td>
    <td><a href="https://huggingface.co/ahmedheakl/cass-sm4090-3b">HuggingFace</a></td>
  </tr>
  <tr>
    <td>CASS-sm4090-7B</td>
    <td>TBR</td>
  </tr>
</table>


## Todos

- [ ] Release training 
- [ ] Release evaluation scripts.


## Citation
If you are using CASS in your research or application, please cite us using this BibTeX:

```bibtex
@article{heakl2025cass,
  title={CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark},
  author={Heakl, Ahmed and Hashmi, Sarim and Stahl, Gustavo Bertolo and Han, Seung Hun Eddie and Khan, Salman and Mahmoud, Abdulrahman},
  journal={arXiv preprint arXiv:2505.16968},
  year={2025}
}
```