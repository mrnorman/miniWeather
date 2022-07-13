#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/11.1.0 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="jsrun -n 1 -a 1 -c 1 -g 1"

export OMPI_FC=gfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                     \
      -DOPENMP_FLAGS="-fopenmp"                                           \
      -DOPENACC_FLAGS="-fopenacc -foffload=nvptx-none=\"-lm -Ofast -ffast-math -moptimize\" -fopenacc-dim=16384:1:128 -fopt-info-omp" \
      -DOPENMP45_FLAGS="-fopenmp -foffload=nvptx-none=\"-lm -Ofast -latomic -ffast-math -moptimize\" -fopt-info-omp"                  \
      -DFFLAGS="-DSINGLE_PREC -Ofast -ffast-math -mcpu=native -mtune=native -ffree-line-length-none -DNO_INFORM -I${OLCF_PARALLEL_NETCDF_ROOT}/include"  \
      -DLDFLAGS="-L -I${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                   \
      -DNX=2048                                                           \
      -DNZ=1024                                                           \
      -DSIM_TIME=10                                                     \
      -DOUT_FREQ=2000                                                   \
      ..

#      -DOPENACC_FLAGS="-fopenacc -fopenacc-dim=:1:128"                                         \
