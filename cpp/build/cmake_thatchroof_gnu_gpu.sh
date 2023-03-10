#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load cmake-3.23.2-gcc-11.1.0-kvgnqc6

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS
unset CXX

export OMPI_CXX=g++-11
export OMPI_FC=gfortran-11
export OMPI_CC=gcc-11

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                        \
      -DCMAKE_Fortran_COMPILER=mpif90                    \
      -DCMAKE_C_COMPILER=mpicc                           \
      -DYAKL_ARCH="CUDA"                                 \
      -DYAKL_CUDA_FLAGS="-O3 -DNO_INFORM --use_fast_math -arch sm_86 -ccbin mpic++ -DSINGLE_PREC" \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"  \
      -DNX=2048                                          \
      -DNZ=1024                                          \
      -DSIM_TIME=10                                      \
      -DOUT_FREQ=-1                                      \
      -DYAKL_HAVE_MPI=ON                                 \
      ..

