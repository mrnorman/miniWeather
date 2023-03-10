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
export OMPI_CC=icc

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpicxx                        \
      -DCMAKE_Fortran_COMPILER=mpif90                    \
      -DCMAKE_C_COMPILER=mpicc                           \
      -DYAKL_CXX_FLAGS="-fast -fast-transcendentals -fp-model=fast=2 -fno-alias -DNO_INFORM -DHAVE_MPI -DSIMD_LEN=32"  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"  \
      -DNX=256                                           \
      -DNZ=128                                           \
      -DSIM_TIME=250                                    \
      -DOUT_FREQ=-1                                     \
      ..

