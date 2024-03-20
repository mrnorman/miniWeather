#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu gcc/12.2.0 cray-parallel-netcdf craype-accel-amd-gfx90a

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                                                \
      -DCXXFLAGS="-O3 -DNO_INFORM"                                           \
      -DLDFLAGS=""                                                           \
      -DOPENACC_FLAGS="-O3 -fopenacc -foffload=\"-lm -latomic\" -DNO_INFORM" \
      -DOPENMP_FLAGS="-O3 -fopenmp -DNO_INFORM"                              \
      -DOPENMP45_FLAGS="-O3 -fopenmp -foffload=\"-lm -latomic\" -DNO_INFORM" \
      -DNX=2048                                                              \
      -DNZ=1024                                                              \
      -DDATA_SPEC="DATA_SPEC_THERMAL"                                        \
      -DSIM_TIME=100                                                         \
      -DOUT_FREQ=-1                                                          \
      ..

