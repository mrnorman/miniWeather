#!/bin/bash

module load gcc/10.3.0 parallel-netcdf cmake

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DFFLAGS="-Ofast -march=native -mtune=native -DNO_INFORM"                 \
      -DNX=256 \
      -DNZ=128 \
      -DSIM_TIME=250  \
      -DOUT_FREQ=2000 \
      ..

