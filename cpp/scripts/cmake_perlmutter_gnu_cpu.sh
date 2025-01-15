#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load PrgEnv-gnu gcc/11.2.0 cray-mpich cray-parallel-netcdf cmake

export TEST_MPI_COMMAND="srun -n 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                                                     \
      -DCXXFLAGS="-Ofast -march=native -DNO_INFORM -std=c++11 -I${PNETCDF_DIR}/include"   \
      -DLDFLAGS="-L${PNETCDF_DIR}/lib -lpnetcdf"                        \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      -DOUT_FREQ=2000                     \
      ..
