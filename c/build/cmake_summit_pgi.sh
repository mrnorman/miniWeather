#!/bin/bash

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENMP_FLAGS=-mp                            \
      -DOPENACC_FLAGS="-ta=nvidia,cc70,ptxinfo"     \
      -DCXXFLAGS="-O3"                              \
      -DLDFLAGS=""                                  \
      ..

