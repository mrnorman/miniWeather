#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load PrgEnv-gnu gcc/11.2.0 cray-mpich cray-parallel-netcdf cmake

export TEST_MPI_COMMAND="srun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                     \
      -DCMAKE_C_COMPILER=mpicc                        \
      -DCMAKE_Fortran_COMPILER=mpif90                 \
      -DYAKL_CXX_FLAGS="-O3 -I${PNETCDF_DIR}/include" \
      -DLDFLAGS="-L${PNETCDF_DIR}/lib -lpnetcdf"      \
      -DNX=200                                        \
      -DNZ=100                                        \
      -DSIM_TIME=1000                                 \
      -DOUT_FREQ=2000                                 \
      -DYAKL_HAVE_MPI=ON                              \
      ..

