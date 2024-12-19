#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-amd amd/6.0.0 cray-parallel-netcdf

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"
unset MPICH_GPU_SUPPORT_ENABLED

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                       \
      -DCMAKE_C_COMPILER=cc                         \
      -DCMAKE_Fortran_COMPILER=ftn                  \
      -DYAKL_ARCH=""                                \
      -DYAKL_CXX_FLAGS="-DNO_INFORM -Ofast -march=native" \
      -DLDFLAGS=""                                  \
      -DNX=2048                                     \
      -DNZ=1024                                     \
      -DDATA_SPEC="DATA_SPEC_THERMAL"               \
      -DSIM_TIME=100                                \
      -DOUT_FREQ=-1                                 \
      -DYAKL_HAVE_MPI=ON                            \
      ..

