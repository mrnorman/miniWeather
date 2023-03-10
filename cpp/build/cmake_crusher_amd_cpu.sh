#!/bin/bash

source ${MODULESHOME}/init/bash
module load PrgEnv-amd cray-parallel-netcdf cmake craype-accel-amd-gfx90a

export TEST_MPI_COMMAND=""
unset CXX
unset CC
unset FC
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                       \
      -DCMAKE_C_COMPILER=cc                         \
      -DCMAKE_Fortran_COMPILER=ftn                  \
      -DYAKL_ARCH=""                                \
      -DYAKL_CXX_FLAGS="-DNO_INFORM -DGPU_AWARE_MPI -O3 -ffast-math -I${PNETCDF_DIR}/include" \
      -DLDFLAGS="-L${PNETCDF_DIR}/lib -lpnetcdf"  \
      -DNX=16384                                    \
      -DNZ=8192                                     \
      -DSIM_TIME=0.1                                \
      -DOUT_FREQ=-1                                 \
      -DYAKL_HAVE_MPI=ON                            \
      ..

