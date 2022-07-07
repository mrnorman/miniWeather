#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/11.2.0 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                                                     \
      -DCXXFLAGS="-O3 -std=c++11 -I${OLCF_PARALLEL_NETCDF_ROOT}/include"   \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                        \
      -DOPENACC_FLAGS="-fopenacc -foffload=\"-lm -latomic\""                          \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DOPENMP45_FLAGS="-fopenmp -foffload=\"-lm -latomic\""                          \
      -DNX=200                                                                        \
      -DNZ=100                                                                        \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES"                                           \
      -DSIM_TIME=1000                                                                 \
      ..

