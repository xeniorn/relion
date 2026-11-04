# GPU device identification and MPI GPU-mapping tests

Stand-alone tests (no RELION build needed) for the GPU device-identification fix
(`src/acc/device_id.{h,cpp}`).

| File | What it tests |
|---|---|
| `test_device_id.cpp`, `test_device_id.sbatch` | The `DeviceIdSource` implementations and `DeviceIdResolver`: helpers, ordering, environment lists, and on a GPU node the real CUDA UUIDs (MIG-aware), PCI and index sources. CPU-only and CUDA builds. |
| `relion_gpumap_probe.cpp` | A stand-alone copy of RELION's MPI GPU assignment (`MlOptimiserMpi::initialise`): the old "Device <index> on <host>" identity next to a UUID-based one, with the real sharing measured from device memory, RELION-like allocation and MPI traffic. Needs only an MPI compiler (`libcuda` is loaded at run time). Run it under different Slurm GPU placements, e.g. `--gres=gpu:N` per node vs `--gpus-per-task=1`. |

Finding that motivated the fix: when a scheduler binds one GPU per MPI rank, every rank sees its
GPU as index 0, so ranks on one host were treated as sharing one GPU and each got about half of
its own device's memory.

Build the probe: `mpicxx -O2 -std=c++17 -o relion_gpumap_probe relion_gpumap_probe.cpp -ldl`;
run: `srun --mpi=pmix ./relion_gpumap_probe [--gpu "0:1"] [--j 2] [--alloc]` (rank 0 is the leader).
