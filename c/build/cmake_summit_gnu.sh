#!/bin/bash

##############################################################################
## This requires gcc/8.1.1 on Summit right now
##############################################################################

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/9.3.0 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENACC_FLAGS="-fopenacc -foffload=\"-lm -O3\"" \
      -DOPENMP_FLAGS="-fopenmp"                     \
      -DOPENMP45_FLAGS="-fopenmp -foffload=\"-lm -O3\"" \
      -DCXXFLAGS="-O3"                              \
      -DLDFLAGS=""                                  \
      -DNX=200 \
      -DNZ=100 \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES" \
      -DSIM_TIME=1000 \
      ..

