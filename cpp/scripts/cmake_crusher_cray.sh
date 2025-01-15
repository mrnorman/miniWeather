#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-cray-amd cce/17.0.0 cray-parallel-netcdf craype-accel-amd-gfx90a amd-mixed

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"
unset MPICH_GPU_SUPPORT_ENABLED

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                     \
      -DCXXFLAGS="-Ofast -march=native -DNO_INFORM"                \
      -DLDFLAGS=""                                \
      -DOPENMP_FLAGS="-Ofast -march=native -fopenmp -DNO_INFORM"   \
      -DOPENMP45_FLAGS="-Ofast -march=native -fopenmp -DNO_INFORM" \
      -DNX=2048                                   \
      -DNZ=1024                                   \
      -DDATA_SPEC="DATA_SPEC_THERMAL"             \
      -DSIM_TIME=100                              \
      -DOUT_FREQ=-1                               \
      ..

