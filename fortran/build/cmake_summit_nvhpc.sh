#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps nvhpc/22.5 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                                                  \
      -DFFLAGS="-O3 -Mvect -Mextend -DNO_INFORM -I${OLCF_PARALLEL_NETCDF_ROOT}/include"                \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                                         \
      -DOPENMP_FLAGS="-Minfo=mp"                                                                       \
      -DOPENACC_FLAGS:STRING="-acc -gpu=cc70,fastmath,loadcache:L1,ptxinfo -Minfo=accel"               \
      -DOPENMP45_FLAGS="-Minfo=mp -mp=gpu -gpu=cc70,fastmath,loadcache:L1,ptxinfo"                     \
      -DDO_CONCURRENT_FLAGS:STRING="-stdpar=gpu -Minfo=stdpar -gpu=cc70,fastmath,loadcache:L1,ptxinfo" \
      -DNX=200                                                                                         \
      -DNZ=100                                                                                         \
      -DSIM_TIME=1000                                                                                  \
      -DOUT_FREQ=2000                                                                                  \
      ..
