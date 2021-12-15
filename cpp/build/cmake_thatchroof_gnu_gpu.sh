#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${PNETCDF_PATH}                \
      -DYAKL_CUDA_FLAGS="-O3 -DHAVE_MPI --use_fast_math -arch sm_35 -ccbin mpic++"      \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES"         \
      -DSIM_TIME=1000                               \
      -DYAKL_ARCH="CUDA"                            \
      ..

