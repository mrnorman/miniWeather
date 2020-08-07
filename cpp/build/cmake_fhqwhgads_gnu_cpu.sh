#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${PNETCDF_PATH}                \
      -DCXXFLAGS="-O3 -std=c++11"                   \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=5 \
      ..

