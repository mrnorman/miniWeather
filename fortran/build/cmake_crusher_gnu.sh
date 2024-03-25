#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu gcc/12.2.0 cray-parallel-netcdf craype-accel-amd-gfx90a rocm

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"
unset MPICH_GPU_SUPPORT_ENABLED


./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=ftn                                                       \
      -DFFLAGS="-Ofast -march=native -DNO_INFORM -ffree-line-length-none"                \
      -DLDFLAGS=""                                                                       \
      -DOPENMP_FLAGS="-Ofast -march=native -fopenmp -DNO_INFORM -ffree-line-length-none" \
      -DNX=2048                                                                          \
      -DNZ=1024                                                                          \
      -DDATA_SPEC="DATA_SPEC_THERMAL"                                                    \
      -DSIM_TIME=100                                                                     \
      -DOUT_FREQ=-1                                                                      \
      ..


