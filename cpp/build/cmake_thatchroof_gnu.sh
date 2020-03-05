#!/bin/bash

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${PNETCDF_PATH}                \
      -DYAKL_CUB_HOME=`pwd`/../cub                  \
      -DCXXFLAGS="-O3 -std=c++14"                   \
      -DCUDA_FLAGS="-arch sm_35 -ccbin mpic++"   \
      -DNX=2000 \
      -DNZ=1000 \
      -DSIM_TIME=5 \
      ..

#     -DARCH="CUDA"                                 \
