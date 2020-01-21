#!/bin/bash

##############################################################################
## This requires gcc/8.1.1 on Summit right now
##############################################################################

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENACC_FLAGS="-fopenacc -foffload=\"-lm -O3\" -fopenacc-dim=-:1:128" \
      -DOPENMP_FLAGS="-fopenmp"                     \
      -DOPENMP45_FLAGS="-fopenmp -foffload=\"-lm -O3\" -fopenacc-dim=-:1:128" \
      -DFFLAGS="-O3"                                \
      -DLDFLAGS=""                                  \
      -DNX=2000 \
      -DNZ=1000 \
      -DSIM_TIME=5 \
      ..

