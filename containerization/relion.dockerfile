

# clip cluster Nvidia driver 535.129 only compatible up to CUDA 12.2
# ARG DOCKER_CUDA_IMAGE_VERSION="12.3.2-cudnn9-devel-ubuntu22.04"
ARG DOCKER_CUDA_IMAGE_VERSION="12.2.2-cudnn8-devel-ubuntu20.04"
ARG CONTAINER_ENV_VARS_FILE="/config/docker-env-vars.env"

ARG CONTAINER_DATA_MOUNT_ROOT="/data"
ARG CONTAINER_TORCH_HOME="${CONTAINER_DATA_MOUNT_ROOT}/torch/home"
ARG CONTAINER_RELION_SCRATCH_DIR="/scratch"

ARG OMPI_SLURM_SUPPORT=1
ARG OMPI_VERSION="3.1.1"
ARG OMPI_CONFIG_ARGS=""
ARG CONTAINER_OMPI_DIR="/opt/ompi"

ARG CONDA_PACKAGE="miniforge3-22.9.0-3"
ARG CONTAINER_PYENV_DIR="/usr/local/apps/pyenv"
ARG CONTAINER_CONDA_DIR="${CONTAINER_PYENV_DIR}/versions/${CONDA_PACKAGE}"
ARG CONTAINER_RELION_ENV_NAME="relion_env"

ARG CTFFIND_VERSION="4.1.14"
ARG CONTAINER_CTFFIND_DIR="/usr/local/apps/ctffind-${CTFFIND_VERSION}"

ARG RELION_CUDA_ARCH="50"
ARG RELION_CUDA_ON_OFF="ON"
ARG CONTAINER_RELION_INSTALL_DIR="/usr/local/apps/relion"
ARG CONTAINER_RELION_CONDA_DIR="${CONTAINER_CONDA_DIR}/envs/${CONTAINER_RELION_ENV_NAME}"

ARG RELIONAPP_REPO_URL="https://github.com/xeniorn/relion.git"
ARG RELIONAPP_REPO_BRANCH="devel"
# unset by default, take the default from the selected git branch
ARG RELIONAPP_REPO_COMMIT

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

RUN apt-get install -y cmake git curl

RUN apt-get install -y \
    build-essential \
    gcc-9 g++-9 \
    libtiff-dev libpng-dev ghostscript libxft-dev libgl1-mesa-dev

# Default to gcc-9 and g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 99 \
&& update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 99

##############################################################################################################################################################
FROM build as build_mpi

RUN apt-get install -y wget git bash gfortran make file bzip2

ARG OMPI_SLURM_SUPPORT
RUN if [ ${OMPI_SLURM_SUPPORT} -eq 1 ]; then apt-get install -y libpmi2-0-dev; fi

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
    && rm "${ompi_archive_filename}" \
    && cd "${ompi_package_foldername}" \
    && export ompi_config_args="--prefix=${CONTAINER_OMPI_DIR} --with-cuda ${OMPI_CONFIG_ARGS}" \
    && if [ ${OMPI_SLURM_SUPPORT} -eq 1 ]; then export ompi_config_args="${ompi_config_args} --with-slurm --with-pmi"; fi \
    && ./configure ${ompi_config_args} \
    && make install

ARG CONTAINER_OMPI_DIR
ENV PATH=${CONTAINER_OMPI_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CONTAINER_OMPI_DIR}/lib:${LD_LIBRARY_PATH}
ENV MANPATH=${CONTAINER_OMPI_DIR}/share/man:${MANPATH}

ARG CONTAINER_ENV_VARS_FILE
RUN mkdir -p $(dirname "${CONTAINER_ENV_VARS_FILE}") \
    && echo "export CONTAINER_OMPI_DIR=${CONTAINER_OMPI_DIR}" >>${CONTAINER_ENV_VARS_FILE} \
    && echo "export PATH=${CONTAINER_OMPI_DIR}/bin:\${PATH}" >>${CONTAINER_ENV_VARS_FILE} \
    && echo "export LD_LIBRARY_PATH=${CONTAINER_OMPI_DIR}/lib:\${LD_LIBRARY_PATH}" >>${CONTAINER_ENV_VARS_FILE} \
    && echo "export MANPATH=${CONTAINER_OMPI_DIR}/share/man:\${MANPATH}" >>${CONTAINER_ENV_VARS_FILE} 

RUN rm -rf ${TEMP_OMPI_BUILD_DIR}

##############################################################################################################################################################
FROM build_mpi as setup_conda_env
### Install python libraries for RELION                       ###
### Blush, DynaMight, Model-Angelo, Classranker, Topaz, etc.) ###
# Install Pyenv to /usr/local/apps
ARG CONTAINER_ENV_VARS_FILE

ARG CONTAINER_PYENV_DIR

RUN git clone https://github.com/yyuu/pyenv.git ${CONTAINER_PYENV_DIR}

ENV PATH="${CONTAINER_PYENV_DIR}/bin:${PATH}"
# required, don't remove PYENV_ROOT!
ENV PYENV_ROOT=${CONTAINER_PYENV_DIR}
RUN echo "export PATH=${CONTAINER_PYENV_DIR}/bin:\${PATH}" >>${CONTAINER_ENV_VARS_FILE}

ARG CONDA_PACKAGE

# Install Miniforge through Pyenv
RUN pyenv install --list \
&& pyenv install ${CONDA_PACKAGE} \
&& pyenv global ${CONDA_PACKAGE} \
&& pyenv versions

# Activate the environment of installed Miniforge
ARG CONTAINER_CONDA_DIR
ENV PATH="${CONTAINER_CONDA_DIR}/bin:${PATH}"
RUN echo "export PATH=${CONTAINER_CONDA_DIR}/bin:\${PATH}" >>${CONTAINER_ENV_VARS_FILE}

# Update conda
RUN conda update -n base conda

# Clone RELION (ver. 5.0-beta) repository to /usr/local/apps/relion-git
ARG TEMP_GIT_DIR="/tmp/relion-git/build_conda"
ARG RELIONAPP_REPO_URL
ARG RELIONAPP_REPO_BRANCH
RUN git clone ${RELIONAPP_REPO_URL} -b ${RELIONAPP_REPO_BRANCH} ${TEMP_GIT_DIR}
WORKDIR ${TEMP_GIT_DIR}

ARG RELIONAPP_REPO_COMMIT
RUN if [ ! -v RELIONAPP_REPO_COMMIT ]; then \
ENV RELIONAPP_REPO_COMMIT=$(git rev-parse HEAD); \
fi \
&& git checkout ${RELIONAPP_REPO_COMMIT}

ARG CONTAINER_RELION_ENV_NAME
RUN sed -i -r -e 's|name: .+|name: ${CONTAINER_RELION_ENV_NAME}|g' ./environment.yml

# Create a conda environment for RELION (relion-conda)
RUN conda env create -f ./environment.yml

# Activate the relion-conda
ARG CONTAINER_RELION_CONDA_DIR

ENV PATH="${CONTAINER_RELION_CONDA_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CONTAINER_RELION_CONDA_DIR}/lib:${LD_LIBRARY_PATH}"

RUN echo "export PATH=${CONTAINER_RELION_CONDA_DIR}/bin:\${PATH}" >>${CONTAINER_ENV_VARS_FILE} \
    && echo "export LD_LIBRARY_PATH=${CONTAINER_RELION_CONDA_DIR}/lib:\${LD_LIBRARY_PATH}" >>${CONTAINER_ENV_VARS_FILE}

RUN rm -rf ${TEMP_GIT_DIR}

##############################################################################################################################################################
FROM setup_conda_env as build_relion

# break cache
# RUN mkdir -p /tmp && date >/tmp/date_hehe

ARG TEMP_GIT_DIR="/tmp/relion-git/cpp/repo"
RUN git clone ${RELIONAPP_REPO_URL} -b ${RELIONAPP_REPO_BRANCH} ${TEMP_GIT_DIR}

### Install RELION (ver. 5.0-beta) ###
# Prepare directories for RELION build
ARG TEMP_BUILD_DIR="/tmp/relion-git/cpp/build"
RUN mkdir -p ${TEMP_BUILD_DIR}

ARG CONTAINER_TORCH_HOME
ENV CONTAINER_TORCH_HOME=${CONTAINER_TORCH_HOME}

#RUN mkdir -p ${CONTAINER_TORCH_HOME} \
#    && cp -r /usr/local/apps/relion-git/containerization/fake_data/* ${CONTAINER_TORCH_HOME}/

WORKDIR ${TEMP_BUILD_DIR}

ARG CONTAINER_RELION_CONDA_DIR
ARG CONTAINER_RELION_INSTALL_DIR
ARG RELION_CUDA_ARCH
ARG RELION_CUDA_ON_OFF

# Install RELION to /usr/local/apps/relion-v5.0-beta
# Add -DAMDFFTW=ON to the following (if AMD CPU)
RUN cmake \
-DCMAKE_INSTALL_PREFIX="${CONTAINER_RELION_INSTALL_DIR}" \
-DFORCE_OWN_FFTW=ON -DFORCE_OWN_FLTK=ON \
-DPYTHON_EXE_PATH="${CONTAINER_RELION_CONDA_DIR}/bin/python" \
-DTORCH_HOME_PATH="${CONTAINER_TORCH_HOME}" \
-DCMAKE_CXX_FLAGS="-pthread"  -DDoublePrec_GPU=OFF -DDoublePrec_CPU=ON  \
-DCMAKE_SHARED_LINKER_FLAGS="-lpthread" \
-DCUDA_ARCH=${RELION_CUDA_ARCH} \
-DCUDA=${RELION_CUDA_ON_OFF} \
-DFETCH_WEIGHTS=OFF \
${TEMP_GIT_DIR} \
&& make \
&& make install

RUN rm -rf ${TEMP_BUILD_DIR} ${TEMP_GIT_DIR}

ENTRYPOINT ["/bin/bash", "-c"]

##############################################################################################################################################################
FROM build_mpi as build_ctffind

### Install CTFFIND-4.1.14 ###
# Prepare the installation directory
ARG CTFFIND_VERSION
ARG CONTAINER_CTFFIND_DIR

RUN mkdir -p ${CONTAINER_CTFFIND_DIR}
WORKDIR ${CONTAINER_CTFFIND_DIR}

ARG TEMP_CTFFIND_ARCHIVE="ctffind-${CTFFIND_VERSION}-linux64.tar.gz"

# Download CTFFIND
ARG DL_URL="https://grigoriefflab.umassmed.edu/system/tdf?path=${TEMP_CTFFIND_ARCHIVE}&file=1&type=node&id=26"
RUN curl -L ${DL_URL} -o ${TEMP_CTFFIND_ARCHIVE}

# Extract and then remove the downloaded .tar.gz file
RUN tar -xvf ${TEMP_CTFFIND_ARCHIVE} \
    && rm -f ${TEMP_CTFFIND_ARCHIVE}



##############################################################################################################################################################
FROM build_relion as clean

# Clean up apt
RUN apt-get autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Clean up conda & pip
RUN conda clean --all --force-pkgs-dirs --yes \
&& pip cache purge


##############################################################################################################################################################
FROM scratch as final

COPY --from=clean / /
COPY --from=build_ctffind ${CONTAINER_CTFFIND_DIR} ${CONTAINER_CTFFIND_DIR}

#RUN cat ${CONTAINER_ENV_VARS_FILE} >>~/.bashrc

# For OpenMPI
ARG CONTAINER_OMPI_DIR
ENV PATH=${CONTAINER_OMPI_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CONTAINER_OMPI_DIR}/lib:${LD_LIBRARY_PATH}
ENV MANPATH=${CONTAINER_OMPI_DIR}/share/man:${MANPATH}

# For RELION
ARG CONTAINER_RELION_INSTALL_DIR
ENV PATH="${CONTAINER_RELION_INSTALL_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CONTAINER_RELION_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"

# For Blush, DynaMight, Model-Angelo, Classranker, Topaz, etc.
ARG CONTAINER_TORCH_HOME
ENV CONTAINER_TORCH_HOME=${CONTAINER_TORCH_HOME}

ARG CONTAINER_PYENV_DIR
ENV PATH="${CONTAINER_PYENV_DIR}/bin:${PATH}"

ARG CONTAINER_CONDA_DIR="${CONTAINER_PYENV_DIR}/versions/miniforge3-22.9.0-3"
ENV PATH="${CONTAINER_CONDA_DIR}/bin:${PATH}"

ARG CONTAINER_RELION_CONDA_DIR
ENV PATH="${CONTAINER_RELION_CONDA_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CONTAINER_RELION_CONDA_DIR}/lib:${LD_LIBRARY_PATH}"

# Default CTFFIND-4.1+ executable
ARG CONTAINER_CTFFIND_DIR
ENV RELION_CTFFIND_EXECUTABLE="${CONTAINER_CTFFIND_DIR}/bin/ctffind"

# The default scratch directory in the GUI
# (depends on your environment outside this container)
ARG CONTAINER_RELION_SCRATCH_DIR
RUN mkdir -p ${CONTAINER_RELION_SCRATCH_DIR}
ENV RELION_SCRATCH_DIR="${CONTAINER_RELION_SCRATCH_DIR}"

ENTRYPOINT ["/bin/bash", "-c"]