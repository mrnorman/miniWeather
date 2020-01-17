#!/bin/bash

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENMP_FLAGS="-qsmp=omp"                    \
      -DOPENMP45_FLAGS="-qsmp=omp -qoffload"        \
      -DFFLAGS="-O3"                                \
      -DLDFLAGS=""                                  \
      ..

