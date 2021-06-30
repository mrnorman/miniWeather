#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${PNETCDF_PATH}                \
      -DYAKL_CUDA_FLAGS="-O3 --use_fast_math -arch sm_50 -ccbin mpic++"      \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      -DYAKL_ARCH="CUDA"                            \
      ..

