#!/bin/bash

module load gcc/10.3.0 parallel-netcdf cmake

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_CXX=g++
export OMPI_CC=gcc
export OMPI_F90=gfortran

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++         \
      -DCMAKE_C_COMPILER=mpicc            \
      -DCMAKE_Fortran_COMPILER=mpif90     \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}      \
      -DYAKL_CXX_FLAGS="-Ofast -std=c++11 -DNO_INFORM"   \
      -DNX=200                            \
      -DNZ=100                            \
      -DSIM_TIME=1000                     \
      -DOUT_FREQ=2000 \
      ..

