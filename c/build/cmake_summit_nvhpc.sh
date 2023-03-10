#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps nvhpc/22.11 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                                                                      \
      -DCXXFLAGS="-O3 -Mvect -DNO_INFORM -std=c++11 -I${OLCF_PARALLEL_NETCDF_ROOT}/include"                \
      -DOPENMP_FLAGS="-mp -Minfo=mp"                                                                   \
      -DOPENACC_FLAGS:STRING="-acc -gpu=cc70,fastmath,loadcache:L1,ptxinfo -Minfo=accel"               \
      -DOPENMP45_FLAGS="-Minfo=mp -mp=gpu -gpu=cc70,fastmath,loadcache:L1,ptxinfo"                     \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                                         \
      -DNX=200                                                                                         \
      -DNZ=100                                                                                         \
      -DSIM_TIME=1000                                                                                  \
      -DOUT_FREQ=2000                                                                                  \
      ..
