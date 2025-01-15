#!/bin/bash

source /usr/share/modules/init/bash
module purge

export TEST_MPI_COMMAND="mpirun -n 1"

export OMPI_FC=gfortran-11

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                     \
      -DFFLAGS="-Ofast -march=native -ffree-line-length-none -DNO_INFORM -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"                   \
      -DOPENMP_FLAGS="-fopenmp"                                           \
      -DOPENACC_FLAGS="-fopenacc"                                         \
      -DOPENMP45_FLAGS="-fopenmp"                                         \
      -DNX=200                                                            \
      -DNZ=100                                                            \
      -DSIM_TIME=1000                                                     \
      -DOUT_FREQ=1000                                                     \
      ..

