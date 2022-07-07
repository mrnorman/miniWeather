#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load icc

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS
unset CXX

export OMPI_CXX=icpc
export OMPI_FC=ifort
export OMPI_F90=ifort
export OMPI_CC=icc

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpicxx                        \
      -DCMAKE_Fortran_COMPILER=mpif90                    \
      -DCMAKE_C_COMPILER=mpicc                           \
      -DYAKL_CXX_FLAGS="-Ofast -march=native -mtune=native -DNO_INFORM -DHAVE_MPI -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"  \
      -DNX=200                                           \
      -DNZ=100                                           \
      -DSIM_TIME=1000                                    \
      -DOUT_FREQ=1000                                    \
      ..

