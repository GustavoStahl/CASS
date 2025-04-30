#!/bin/bash
#NOTE: This file is made to be run inside the docker environment
./hipify-clang --clang-resource-directory /usr/lib/llvm-18/lib/clang/18 --cuda-gpu-arch sm_80 -o sample.hip sample.cu