#!/usr/bin/env bash
# Compile and run test_acc_device_id.cpp against the real CUDA toolkit (NVIDIA devel image) on a GPU node.
# Env: SRC = relion checkout containing src/acc/acc_device_id.h and tests/gpumap/
set -uo pipefail
W=$SCRATCHDIR/claude/cudadev; mkdir -p $W/tmp; export APPTAINER_TMPDIR=$W/tmp APPTAINER_CACHEDIR=$W/cache
[ -f $W/cuda-devel.sif ] || apptainer pull --disable-cache $W/cuda-devel.sif docker://docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
cd "$SRC"
apptainer exec --nv $W/cuda-devel.sif nvcc -x cu -D_CUDA_ENABLED -I. -o $W/t tests/gpumap/test_acc_device_id.cpp -lcuda && apptainer exec --nv $W/cuda-devel.sif $W/t
echo "--- nvidia-smi -L"; nvidia-smi -L
