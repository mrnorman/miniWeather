#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${PNETCDF_PATH}   \
      -DFFLAGS="-Ofast -DNO_INFORM -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"                                \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      -DOUT_FREQ=1000 \
      ..

