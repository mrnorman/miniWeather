#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++         \
      -DPNETCDF_PATH=${PNETCDF_PATH}      \
      -DYAKL_CXX_FLAGS="-O3 -std=c++11"   \
      -DNX=200                            \
      -DNZ=100                            \
      -DSIM_TIME=1000                     \
      ..

