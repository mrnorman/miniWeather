#!/bin/bash

##############################################################################
## This requires gcc/8.1.1 on Summit right now
##############################################################################

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DCXXFLAGS="-O3"                              \
      -DARCH="CUDA"                                 \
      -DCUDA_FLAGS="-arch sm_70"                     \
      -DLDFLAGS=""                                  \
      ..

