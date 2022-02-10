#!/bin/bash

source /usr/share/modules/init/bash
module purge

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_FC=gfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${PNETCDF_PATH}   \
      -DFFLAGS="-Ofast -march=native -mtune=native -DNO_INFORM -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"                                \
      -DNX=256 \
      -DNZ=128 \
      -DSIM_TIME=250  \
      -DOUT_FREQ=2000 \
      ..

