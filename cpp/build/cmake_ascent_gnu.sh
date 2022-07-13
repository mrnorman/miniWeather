#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/9.3.0 cuda/11.4.2 parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DCMAKE_C_COMPILER=mpicc                      \
      -DCMAKE_Fortran_COMPILER=mpif90               \
      -DYAKL_ARCH="CUDA"                            \
      -DYAKL_CUDA_FLAGS="-DSINGLE_PREC -DNO_INFORM -DHAVE_MPI -O3 --use_fast_math -arch sm_70 -ccbin mpic++ -I${OLCF_PARALLEL_NETCDF_ROOT}/include" \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"  \
      -DNX=2048                                     \
      -DNZ=1024                                     \
      -DSIM_TIME=10                                 \
      -DOUT_FREQ=-1                                 \
      ..

