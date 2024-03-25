#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-cray cray-parallel-netcdf cmake craype-accel-amd-gfx90a

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"
export MPICH_GPU_SUPPORT_ENABLED=1

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                       \
      -DCMAKE_C_COMPILER=cc                         \
      -DCMAKE_Fortran_COMPILER=ftn                  \
      -DYAKL_ARCH=""                                \
      -DYAKL_CXX_FLAGS="-DNO_INFORM -DGPU_AWARE_MPI -O3" \
      -DLDFLAGS=""                                  \
      -DNX=2048                                     \
      -DNZ=1024                                     \
      -DDATA_SPEC="DATA_SPEC_THERMAL"               \
      -DSIM_TIME=100                                \
      -DOUT_FREQ=-1                                 \
      ..

