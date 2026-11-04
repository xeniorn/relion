#!/usr/bin/env bash
# Compile and run test_device_id.cpp against the real CUDA toolkit (NVIDIA devel image) on a GPU node,
# and without CUDA (plain g++ from the same image). Env: SRC = relion checkout.
set -uo pipefail
W=$SCRATCHDIR/claude/cudadev; mkdir -p $W/tmp; export APPTAINER_TMPDIR=$W/tmp APPTAINER_CACHEDIR=$W/cache
[ -f $W/cuda-devel.sif ] || apptainer pull --disable-cache $W/cuda-devel.sif docker://docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
cd "$SRC"
I="apptainer exec --nv $W/cuda-devel.sif"
echo "=== no-GPU build"; $I g++ -std=c++11 -Wall -I. -o $W/t_cpu tests/gpumap/test_device_id.cpp src/acc/device_id.cpp && $I $W/t_cpu
echo "=== CUDA build"; $I nvcc -std=c++17 -D_CUDA_ENABLED -I. -o $W/t_cuda -x cu tests/gpumap/test_device_id.cpp -x cu src/acc/device_id.cpp && $I $W/t_cuda
echo "--- nvidia-smi -L"; nvidia-smi -L
