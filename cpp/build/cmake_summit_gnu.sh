#!/bin/bash

# module load gcc cuda cmake netcdf netcdf-fortran

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DYAKL_CUB_HOME=`pwd`/../cub                  \
      -DCXXFLAGS="-O3 -std=c++11"                   \
      -DARCH="CUDA"                                 \
      -DCUDA_FLAGS="-arch sm_70 -ccbin mpic++"   \
      -DNX=2000 \
      -DNZ=1000 \
      -DSIM_TIME=5 \
      ..

