#!/bin/bash

source ${MODULESHOME}/init/bash
module load PrgEnv-amd cray-parallel-netcdf cmake craype-accel-amd-gfx90a

export TEST_MPI_COMMAND="mpirun -n 1"
unset CXX
unset CC
unset FC
unset CUDAFLAGS
unset CXXFLAGS

export MPICH_GPU_SUPPORT_ENABLED=1

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                       \
      -DCMAKE_C_COMPILER=cc                         \
      -DCMAKE_Fortran_COMPILER=ftn                  \
      -DPNETCDF_PATH=${PNETCDF_DIR}                 \
      -DYAKL_ARCH="HIP"                             \
      -DYAKL_HIP_FLAGS="-DHAVE_MPI -DNO_INFORM -DGPU_AWARE_MPI -O3 -ffast-math -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip" \
      -DCMAKE_EXE_LINKER_FLAGS="--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64" \
      -DNX=1024                                     \
      -DNZ=512                                      \
      -DSIM_TIME=100                               \
      -DOUT_FREQ=-1                                 \
      ..

