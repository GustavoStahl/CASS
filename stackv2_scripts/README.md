# Stack V2 CUDA to HIP

## Instructions
> All the commands below are expected to be executed inside the docker container. ```docker compose run transpiler```

Start by downloading the CUDA files from the Stack.
```shell
python3 write_dataset.py --save-dir <path-to-dir> --num-threads <thread-number>
```

Once you have the files, organize them into their original folder structure according to the repository they were fetched from.

```shell
python3 create_repo_structure.py --source <source-folder> --destination <destination-folder>
```

Clone the repository of the top N repositories in the Stack V2 with the highest count of CUDA files. Compiling the files in their original folder structure should help reducing the amount of errors. 

```shell
# pass as destination the destination folder from the previous step
python3 clone_repos.py --destination <desination-folder> --first-n <how-many-repos-to-clone>
```

Get the host and device assembly for the structured dataset created.
```shell
python3 disassemble_cuda.py --dataset <structured-dataset-path> --sass-dir <output-dir> --arch sm_80
```

Retrieve the source files for the generated assembly.
```shell
python3 grab_assembly_source_from_structured.py --as-dir <directory-with-the-assembly> --structured-dir <structured-dir> --source-dir <output-dir>
```

**Refer to the hipify folder for instruction on how to convert CUDA to HIP**
