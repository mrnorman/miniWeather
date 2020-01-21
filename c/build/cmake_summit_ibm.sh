#!/bin/bash

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENMP_FLAGS="-qsmp=omp"                    \
      -DOPENMP45_FLAGS="-qsmp=omp -qoffload"        \
      -DCXXFLAGS="-O3 -std=c+=11"                   \
      -DLDFLAGS=""                                  \
      -DNX=2000 \
      -DNZ=1000 \
      -DSIM_TIME=5 \
      ..

