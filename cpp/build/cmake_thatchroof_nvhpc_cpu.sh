#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load nvhpc

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS
unset CXX

export OMPI_CXX=nvc++
export OMPI_FC=nvfortran
export OMPI_F90=nvfortran
export OMPI_CC=nvc

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                        \
      -DCMAKE_Fortran_COMPILER=mpif90                    \
      -DCMAKE_C_COMPILER=mpicc                           \
      -DYAKL_CXX_FLAGS="-fastsse -O4 -Mfpapprox -Mfprelaxed -march=native -mtune=native -DNO_INFORM -DHAVE_MPI -DSIMD_LEN=32 -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"  \
      -DNX=256                                           \
      -DNZ=128                                           \
      -DSIM_TIME=250                                     \
      -DOUT_FREQ=-1                                      \
      ..

