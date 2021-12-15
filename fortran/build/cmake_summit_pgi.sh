#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps nvhpc cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${OLCF_PARALLEL_NETCDF_ROOT}   \
      -DOPENMP_FLAGS=-mp                            \
      -DOPENACC_FLAGS="-ta=nvidia,cc70,ptxinfo"     \
      -DDO_CONCURRENT_FLAGS="-stdpar=gpu -Minfo=stdpar -gpu=cc70"     \
      -DFFLAGS="-O3"                                \
      -DLDFLAGS=""                                  \
      -DNX=200 \
      -DNZ=100 \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES" \
      -DSIM_TIME=1000 \
      ..

