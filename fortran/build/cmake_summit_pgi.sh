#!/bin/bash

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENMP_FLAGS=-mp                            \
      -DOPENACC_FLAGS="-ta=nvidia,cc70,ptxinfo"     \
      -DFFLAGS="-O3"                                \
      -DLDFLAGS=""                                  \
      ..

