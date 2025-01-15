#!/bin/bash

module load intel parallel-netcdf cmake

export TEST_MPI_COMMAND="mpirun -n 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                                 \
      -DFFLAGS="-Ofast -march=native -mtune=native -DNO_INFORM -I${OLCF_PARALLEL_NETCDF_ROOT}/include"   \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                        \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DNX=200                                                                        \
      -DNZ=100                                                                        \
      -DSIM_TIME=1000                                                                 \
      -DOUT_FREQ=2000                                                                 \
      ..
