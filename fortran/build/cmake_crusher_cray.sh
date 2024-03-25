#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-cray cce/15.0.1 cray-parallel-netcdf craype-accel-amd-gfx90a amd-mixed

#   cce/14.0.1    cce/15.0.0 (L,D)    cce/15.0.1    cce/16.0.0    cce/16.0.1    cce/17.0.0


export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"
unset MPICH_GPU_SUPPORT_ENABLED

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=ftn                                 \
      -DCMAKE_Fortran_COMPILER_WORKS=1 \
      -DFFLAGS="-Ofast -DNO_INFORM"                  \
      -DLDFLAGS=""   \
      -DOPENMP_FLAGS="-Ofast -h omp -DNO_INFORM"   \
      -DOPENACC_FLAGS="-Ofast -h acc -DNO_INFORM" \
      -DOPENMP45_FLAGS="-Ofast -h omp -DNO_INFORM" \
      -DNX=2048                                                    \
      -DNZ=1024                                                    \
      -DDATA_SPEC="DATA_SPEC_THERMAL"                              \
      -DSIM_TIME=100                                               \
      -DOUT_FREQ=-1                                                \
      ..


