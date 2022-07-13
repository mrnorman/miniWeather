#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps xl cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="mpirun -n 1"

export OMPI_FC=xlf2008_r

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DFFLAGS="-O3 -qhot=fastmath -DNO_INFORM -I${OLCF_PARALLEL_NETCDF_ROOT}/include"           \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                                    \
      -DOPENMP_FLAGS="-qsmp=omp"                               \
      -DOPENMP45_FLAGS="-qsmp=omp -qoffload"                   \
      -DNX=2048 \
      -DNZ=1024 \
      -DSIM_TIME=10   \
      -DOUT_FREQ=2000 \
      ..

