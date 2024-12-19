#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/12.1.0 cuda/12.2.0 cmake parallel-netcdf

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpicxx                                            \
      -DCXXFLAGS="-O3 -DNO_INFORM -I$OLCF_PARALLEL_NETCDF_ROOT/include"                                           \
      -DLDFLAGS="-L$OLCF_PARALLEL_NETCDF_ROOT/lib -lpnetcdf"                 \
      -DOPENACC_FLAGS="-O3 -fopenacc -foffload=\"-lm -latomic -march=sm_70\" -DNO_INFORM -I$OLCF_PARALLEL_NETCDF_ROOT/include" \
      -DOPENMP_FLAGS="-O3 -fopenmp -DNO_INFORM -I$OLCF_PARALLEL_NETCDF_ROOT/include"                              \
      -DOPENMP45_FLAGS="-O3 -fopenmp -foffload=\"-lm -latomic -march=sm_70\" -DNO_INFORM -I$OLCF_PARALLEL_NETCDF_ROOT/include" \
      -DNX=2048                                                              \
      -DNZ=1024                                                              \
      -DDATA_SPEC="DATA_SPEC_THERMAL"                                        \
      -DSIM_TIME=100                                                         \
      -DOUT_FREQ=-1                                                          \
      ..

