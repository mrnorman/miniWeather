#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${PNETCDF_PATH}   \
      -DOPENMP_FLAGS="-fopenmp"                     \
      -DFFLAGS="-O3"                                \
      -DLDFLAGS=""                                  \
      -DNX=2000 \
      -DNZ=1000 \
      -DSIM_TIME=5 \
      ..

