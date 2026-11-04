#!/usr/bin/env bash
# Test build of RELION (this fork) with CUDA 13 on EL10 (nvidia/cuda:*-devel-rockylinux10), on a cluster
# compute node via Apptainer. Not an image build: a writable sandbox of NVIDIA's devel image gets the build
# dependencies (dnf with --fakeroot), then RELION is configured and compiled twice:
#   A) CUDA, sm_100 (B200), no GUI          -> GPU code incl. src/acc/device_id.cpp
#   B) CPU only, with GUI                   -> GUI / job-option code
# Env: W (work dir), REF (git ref, default universal/5.1.1), CUDA_IMAGE, JOBS.
set -uo pipefail
W=${W:?set W}; REF=${REF:-universal/5.1.1}; JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-8}}
CUDA_IMAGE=${CUDA_IMAGE:-docker.io/nvidia/cuda:13.4.3-devel-rockylinux10}
mkdir -p "$W/tmp"; export APPTAINER_TMPDIR=$W/tmp APPTAINER_CACHEDIR=$W/cache
SBX=$W/el10-cuda13-sandbox
t0=$SECONDS; ts() { echo "[$(( (SECONDS-t0)/60 )) min] $*"; }

if [ ! -d "$SBX/usr" ]; then
	ts "unpacking $CUDA_IMAGE into a sandbox"
	apptainer build --sandbox "$SBX" "docker://$CUDA_IMAGE" > "$W/sandbox.log" 2>&1 || { tail -5 "$W/sandbox.log"; exit 1; }
	ts "installing build dependencies (fakeroot)"
	apptainer exec --fakeroot --writable "$SBX" bash -c '
		dnf -y -q install dnf-plugins-core epel-release >/dev/null &&
		dnf config-manager --set-enabled crb &&
		dnf -y -q install git cmake make gcc-c++ python3 openmpi-devel fftw-devel libtiff-devel libpng-devel \
			fltk-devel libX11-devel libXft-devel mesa-libGL-devel zlib-devel >/dev/null' \
		> "$W/deps.log" 2>&1 || { echo "dependency install failed"; tail -20 "$W/deps.log"; exit 1; }
fi
X="apptainer exec -B $W $SBX"
$X sh -c 'echo "OS: $(. /etc/os-release; echo $PRETTY_NAME); $(nvcc --version | tail -2 | head -1); $(gcc --version | head -1); $(cmake --version | head -1)"'

SRC=$W/relion
rm -rf "$SRC"; git clone -q --depth 1 --branch "$REF" https://github.com/xeniorn/relion.git "$SRC" || exit 1
ts "source: $REF $(git -C "$SRC" rev-parse --short HEAD)"

build() { # name, cmake args...
	local name=$1; shift
	local B=$W/build-$name; rm -rf "$B"; mkdir -p "$B"
	ts "configure $name: $*"
	$X bash -c "export PATH=/usr/lib64/openmpi/bin:\$PATH; cd $B && cmake $SRC -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXE_PATH=/usr/bin/python3 -DFETCH_WEIGHTS=OFF $*" > "$B/configure.log" 2>&1 \
		|| { echo "CONFIGURE FAILED ($name)"; tail -25 "$B/configure.log"; return 1; }
	ts "compile $name (-j$JOBS)"
	$X bash -c "export PATH=/usr/lib64/openmpi/bin:\$PATH; cd $B && make -j$JOBS" > "$B/make.log" 2>&1 \
		|| { echo "COMPILE FAILED ($name)"; grep -n -E "error|Error" "$B/make.log" | head -30; return 1; }
	ts "OK $name: $(ls $B/bin | wc -l) programs; warnings in our files: $(grep -c -E 'device_id|pipeline_jobs|gui_jobwindow|gui_entries|maingui' $B/make.log)"
	# a CUDA build must still start without an NVIDIA driver (no DT_NEEDED libcuda), as upstream's does
	echo "libcuda linked: $($X ldd "$B/bin/relion_refine" | grep -c libcuda.so)"
	$X "$B/bin/relion_refine" --version 2>&1 | head -3
}

build cuda13-sm100 -DCUDA=ON -DCUDA_ARCH=100 -DGUI=OFF; ra=$?
build cpu-gui -DCUDA=OFF -DALTCPU=OFF -DGUI=ON; rb=$?
ts "done: cuda13-sm100 rc=$ra, cpu-gui rc=$rb"
