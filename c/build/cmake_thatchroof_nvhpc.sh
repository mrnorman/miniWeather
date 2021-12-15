#!/bin/bash

source /usr/share/modules/init/bash
module use /opt/nvidia/hpc_sdk/modulefiles 
module load nvhpc/21.11

export TEST_MPI_COMMAND="mpirun -n 1 --bind-to none"


./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++               \
      -DPNETCDF_PATH="/opt/parallel-netcdf-1.12.0_nvhpc"   \
      -DOPENMP_FLAGS="-mp"                            \
      -DOPENACC_FLAGS="-ta=nvidia,cc35,ptxinfo"     \
      -DCXXFLAGS="-O3 -DNO_INFORM"                                \
      -DLDFLAGS=""                                  \
      -DNX=200 \
      -DNZ=100 \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES" \
      -DSIM_TIME=1000 \
      ..

