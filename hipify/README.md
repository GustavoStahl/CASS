# Stack V2 CUDA to HIP

## Instructions
> All the commands below are expected to be executed inside the docker container. ```docker compose run transpiler```

Iterate through the structured Stack V2 directory and convert the CUDA files into HIP files. The HIP files will be saved in the same tree-folder structure
```shell
python3 hip_from_repo_structure.py --source <structured-dataset-path> --destination <output-dir> --threads <num-threads>
```