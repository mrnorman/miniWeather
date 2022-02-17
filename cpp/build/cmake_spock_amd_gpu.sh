#!/bin/bash

source ${MODULESHOME}/init/bash
module load PrgEnv-amd cray-parallel-netcdf cmake craype-accel-amd-gfx908

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
      -DYAKL_HIP_FLAGS="-DHAVE_MPI -DNO_INFORM -DGPU_AWARE_MPI -O3 -ffast-math -D__HIP_ROCclr__ -D__HIP_ARCH_GFX908__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx908 -x hip" \
      -DCMAKE_EXE_LINKER_FLAGS="--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64" \
      -DNX=16384                                    \
      -DNZ=8192                                     \
      -DSIM_TIME=1                                 \
      -DOUT_FREQ=-1                                 \
      ..

