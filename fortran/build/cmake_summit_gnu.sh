#!/bin/bash

##############################################################################
## This requires gcc/8.1.1 on Summit right now
##############################################################################

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENACC_FLAGS="-fopenacc"                   \
      -DOPENMP_FLAGS="-fopenmp"                     \
      -DOPENMP45_FLAGS="-fopenmp"                   \
      -DFFLAGS="-O3"                                \
      -DLDFLAGS=""                                  \
      ..

