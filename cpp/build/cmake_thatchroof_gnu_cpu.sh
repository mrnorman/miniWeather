#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load gcc-12.1.0-gcc-11.1.0-g2ai6t2

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS
unset CXX

export OMPI_CXX=g++
export OMPI_FC=gfortran
export OMPI_CC=gcc

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                        \
      -DCMAKE_Fortran_COMPILER=mpif90                    \
      -DCMAKE_C_COMPILER=mpicc                           \
      -DYAKL_CXX_FLAGS="-Ofast -ffast-math -march=native -mtune=native -DNO_INFORM -DHAVE_MPI -DSIMD_LEN=32"  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"  \
      -DNX=256                                           \
      -DNZ=128                                           \
      -DSIM_TIME=250                                     \
      -DOUT_FREQ=1000                                    \
      ..

