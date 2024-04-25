


# clip cluster Nvidia driver 535.129 only compatible up to CUDA 12.2
# ARG DOCKER_CUDA_IMAGE_VERSION="12.3.2-cudnn9-devel-ubuntu22.04"
ARG DOCKER_CUDA_IMAGE_VERSION="12.2.2-cudnn8-devel-ubuntu20.04"

ARG SLURM_SUPPORT=1

ARG OMPI_VERSION="3.1.1"
ARG OMPI_CONFIG_ARGS=""
ARG CONTAINER_OMPI_DIR="/opt/ompi"

##############################################################################################################################################################
FROM nvidia/cuda:${DOCKER_CUDA_IMAGE_VERSION} as build

### Set up Ubuntu ###
# for localtime WARNING
RUN touch /etc/localtime

# Install RELION dependent packages
RUN apt-get update \
    && apt-get upgrade -y

# Add en_US.UTF-8 to locale
RUN apt-get install -y locales \
    && locale-gen en_US.UTF-8

RUN apt-get install -y cmake git curl \
    less nano

RUN apt-get install -y \
    build-essential \
    gcc-9 g++-9

# RUN apt-get install -y openmpi-bin libopenmpi-dev

# Default to gcc-9 and g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 99 \
&& update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 99 \
&& g++ --version \
&& gcc --version

##############################################################################################################################################################
FROM build as build_mpi

RUN apt-get install -y wget git bash gfortran make file bzip2

ARG SLURM_SUPPORT
RUN if [ ${SLURM_SUPPORT} -eq 1 ]; then apt-get install -y libpmi2-0-dev; fi

ARG OMPI_VERSION
ARG CONTAINER_OMPI_DIR

ARG TEMP_OMPI_BUILD_DIR="/tmp/ompi"

RUN mkdir -p ${CONTAINER_OMPI_DIR} \
    && mkdir -p ${TEMP_OMPI_BUILD_DIR}

WORKDIR ${TEMP_OMPI_BUILD_DIR}

ARG OMPI_DOWNLOAD_LINK_BASE="https://download.open-mpi.org/release/open-mpi"
ARG OMPI_CONFIG_ARGS

# download and install openmpi
# https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.1.tar.bz2
RUN echo "Download & install ompi with requested config..." \
    && export ompi_download_base="${OMPI_DOWNLOAD_LINK_BASE}" \
    && export ompi_version="${OMPI_VERSION}" \
    && export ompi_version_no_patch=$(echo "${ompi_version}" | sed -r 's|([0-9]+\.[0-9]+).*|\1|') \
    && export ompi_package_foldername="openmpi-${ompi_version}" \
    && export ompi_archive_filename="${ompi_package_foldername}.tar.bz2" \
    && export ompi_download_url="${ompi_download_base}/v${ompi_version_no_patch}/${ompi_archive_filename}" \
    && cd "${TEMP_OMPI_BUILD_DIR}" \
    && wget "${ompi_download_url}" \
    && tar -xjf "${ompi_archive_filename}" \
    && cd "${ompi_package_foldername}" \
    && export ompi_config_args="--prefix=${CONTAINER_OMPI_DIR} --with-cuda ${OMPI_CONFIG_ARGS}" \
    && if [ ${SLURM_SUPPORT} -eq 1 ]; then export ompi_config_args="${ompi_config_args} --with-slurm --with-pmi"; fi \
    && ./configure ${ompi_config_args} \
    && make install

ENV CONTAINER_OMPI_DIR=${CONTAINER_OMPI_DIR}
ENV PATH=${CONTAINER_OMPI_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CONTAINER_OMPI_DIR}/lib:${LD_LIBRARY_PATH}
ENV MANPATH=${CONTAINER_OMPI_DIR}/share/man:${MANPATH}

##############################################################################################################################################################
FROM build_mpi as test_mpi

RUN mkdir -p /test
WORKDIR /test

RUN printf '/*The Parallel Hello World Program*/\n#include <stdio.h>\n#include <mpi.h>\n\nmain(int argc, char **argv)\n{\n   int node;\n   \n   MPI_Init(&argc,&argv);\n   MPI_Comm_rank(MPI_COMM_WORLD, &node);\n     \n   printf("Hello World from Node %%d\\n",node);\n            \n   MPI_Finalize();\n}' >hello.c
RUN mpicc hello.c -o mpi_hello_world
RUN chmod +rx *


