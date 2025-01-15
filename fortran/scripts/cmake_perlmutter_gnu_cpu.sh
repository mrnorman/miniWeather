#!/bin/bash

source ${MODULESHOME}/init/bash
module load PrgEnv-gnu cray-mpich cray-parallel-netcdf cmake

export TEST_MPI_COMMAND="srun -n 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                                 \
      -DFFLAGS="-Ofast -march=native -ffree-line-length-none -DNO_INFORM -I${PNETCDF_DIR}/include"   \
      -DLDFLAGS="-L${PNETCDF_DIR}/lib -lpnetcdf"                        \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      -DOUT_FREQ=2000                     \
      ..
