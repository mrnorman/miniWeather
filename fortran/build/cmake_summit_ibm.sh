#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps xl cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                          \
      -DFFLAGS="-O3 -I${OLCF_PARALLEL_NETCDF_ROOT}/include"    \
      -DOPENMP_FLAGS="-qsmp=omp"                               \
      -DOPENMP45_FLAGS="-qsmp=omp -qoffload"                   \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf" \
      -DNX=200                                                 \
      -DNZ=100                                                 \
      -DSIM_TIME=1000                                                                 \
      -DOUT_FREQ=2000                                                                 \
      ..
