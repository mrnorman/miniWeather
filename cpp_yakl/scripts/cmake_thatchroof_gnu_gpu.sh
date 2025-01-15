#!/bin/bash

source /usr/share/modules/init/bash
module purge

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS
unset CXX

export OMPI_CXX=g++-11
export OMPI_FC=gfortran-11
export OMPI_F90=gfortran-11
export OMPI_CC=gcc-11

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                       \
      -DCMAKE_Fortran_COMPILER=mpif90                    \
      -DCMAKE_C_COMPILER=mpicc                           \
      -DYAKL_CUDA_FLAGS="-O3 -DHAVE_MPI -DNO_INFORM --use_fast_math -arch sm_86 -ccbin mpic++ -DSINGLE_PREC -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15 --ptxas-options=-v" \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf" \
      -DNX=2048                                         \
      -DNZ=1024                                         \
      -DSIM_TIME=10                                   \
      -DOUT_FREQ=-1                                   \
      -DYAKL_ARCH="CUDA"                                \
      ..

