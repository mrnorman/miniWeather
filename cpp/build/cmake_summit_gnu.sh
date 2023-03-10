#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/9.3.0 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DCMAKE_C_COMPILER=mpicc                      \
      -DCMAKE_Fortran_COMPILER=mpif90               \
      -DYAKL_ARCH="CUDA"                            \
      -DYAKL_CUDA_FLAGS="-O3 --use_fast_math -arch sm_70 -ccbin mpic++ -I${OLCF_PARALLEL_NETCDF_ROOT}/include" \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"  \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      -DOUT_FREQ=2000                               \
      -DYAKL_HAVE_MPI=ON                            \
      ..

