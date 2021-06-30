#!/bin/bash

source ${MODULESHOME}/init/bash
module load rocm openmpi cmake
export OMPI_CXX=hipcc

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=/ccs/home/imn/pnetcdf_hip \
      -DYAKL_ARCH="HIP"                             \
      -DYAKL_HIP_FLAGS="-O3"                        \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      ..

