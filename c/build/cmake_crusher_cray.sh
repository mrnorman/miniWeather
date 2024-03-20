#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-cray-amd cce/17.0.0 cray-parallel-netcdf craype-accel-amd-gfx90a

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                     \
      -DCXXFLAGS="-O3 -DNO_INFORM"                \
      -DLDFLAGS=""                                \
      -DOPENMP_FLAGS="-O3 -fopenmp -DNO_INFORM"   \
      -DOPENMP45_FLAGS="-O3 -fopenmp -DNO_INFORM" \
      -DNX=2048                                   \
      -DNZ=1024                                   \
      -DDATA_SPEC="DATA_SPEC_THERMAL"             \
      -DSIM_TIME=100                              \
      -DOUT_FREQ=-1                               \
      ..

