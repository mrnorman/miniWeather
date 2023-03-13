#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load PrgEnv-nvidia cray-mpich cray-parallel-netcdf cmake cudatoolkit

export TEST_MPI_COMMAND="srun -n 1 -c 1 -a 1 -g 1"

#unset CUDAFLAGS
#unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=CC                       \
      -DCMAKE_C_COMPILER=mpicc                      \
      -DYAKL_ARCH="CUDA"                            \
      -DYAKL_CUDA_FLAGS="-DHAVE_MPI -O3 --use_fast_math -arch sm_70 -ccbin mpic++ -I${PNETCDF_DIR}/include" \
      -DLDFLAGS="-L${PNETCDF_DIR}/lib -lpnetcdf"  \
      -DCXXFLAGS="-I${PNETCDF_DIR}/include -L${PNETCDF_DIR}/lib -lpnetcdf" \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      -DOUT_FREQ=2000                     \
      ..

