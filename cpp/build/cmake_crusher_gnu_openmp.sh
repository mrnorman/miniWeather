#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu gcc/12.2.0 cray-parallel-netcdf

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"
unset MPICH_GPU_SUPPORT_ENABLED

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                        \
      -DCMAKE_C_COMPILER=cc                          \
      -DCMAKE_Fortran_COMPILER=ftn                   \
      -DYAKL_ARCH="OPENMP"                           \
      -DYAKL_OPENMP_FLAGS="-DNO_INFORM -Ofast -march=native -fopenmp" \
      -DLDFLAGS="-fopenmp"                           \
      -DNX=2048                                      \
      -DNZ=1024                                      \
      -DDATA_SPEC="DATA_SPEC_THERMAL"                \
      -DSIM_TIME=100                                 \
      -DOUT_FREQ=-1                                  \
      -DYAKL_HAVE_MPI=ON                             \
      ..

