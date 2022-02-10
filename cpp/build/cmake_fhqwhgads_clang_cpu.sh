#!/bin/bash

source /usr/share/modules/init/bash
module purge

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_CXX=clang++
export OMPI_CC=clang
export OMPI_F90=gfortran



./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++         \
      -DCMAKE_C_COMPILER=mpicc            \
      -DCMAKE_Fortran_COMPILER=mpif90     \
      -DPNETCDF_PATH=${PNETCDF_PATH}      \
      -DYAKL_CXX_FLAGS="-Ofast -march=native -mtune=native -std=c++11 -DNO_INFORM"   \
      -DNX=256                            \
      -DNZ=128                            \
      -DSIM_TIME=250                      \
      -DOUT_FREQ=2000 \
      ..

