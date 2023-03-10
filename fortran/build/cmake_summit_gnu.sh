#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/12.1.0 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                                 \
      -DFFLAGS="-O3 -ffree-line-length-none -I${OLCF_PARALLEL_NETCDF_ROOT}/include"   \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DOPENACC_FLAGS="-fopenacc -foffload=nvptx-none=\"-lm -Ofast -ffast-math -march=sm_70 -moptimize\" -fopenacc-dim=16384:1:128 -fopt-info-omp" \
      -DOPENMP45_FLAGS="-fopenmp -foffload=nvptx-none=\"-lm -Ofast -latomic -ffast-math -march=sm_70 -moptimize\" -fopt-info-omp"                  \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                        \
      -DNX=200                                                                        \
      -DNZ=100                                                                        \
      -DSIM_TIME=1000                                                                 \
      -DOUT_FREQ=2000                                                                 \
      ..

