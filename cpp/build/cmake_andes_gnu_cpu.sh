#!/bin/bash

module load gcc/10.3.0 parallel-netcdf cmake

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

unset OMPI_CXX
unset OMPI_CC
unset OMPI_F90
unset OMPI_FC

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++         \
      -DCMAKE_C_COMPILER=mpicc            \
      -DCMAKE_Fortran_COMPILER=mpif90     \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}      \
      -DYAKL_CXX_FLAGS="-DSIMD_LEN=4 -Ofast -march=native -mtune=native -std=c++11 -DNO_INFORM"   \
      -DNX=256                            \
      -DNZ=128                            \
      -DSIM_TIME=250                      \
      -DOUT_FREQ=2000                     \
      ..

