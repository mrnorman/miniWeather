#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${PNETCDF_PATH}   \
      -DOPENMP_FLAGS="-fopenmp"                     \
      -DCXXFLAGS="-O3 -DNO_INFORM"                                \
      -DLDFLAGS=""                                  \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      -DOUT_FREQ=1000 \
      ..

