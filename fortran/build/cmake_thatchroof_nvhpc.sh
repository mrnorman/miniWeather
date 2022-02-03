#!/bin/bash

source /usr/share/modules/init/bash
module use /opt/nvidia/hpc_sdk/modulefiles 
module load nvhpc/21.11

export TEST_MPI_COMMAND="mpirun -n 1"


./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH="/opt/parallel-netcdf-1.12.0_nvhpc"   \
      -DOPENMP_FLAGS="-mp"                            \
      -DOPENACC_FLAGS="-ta=nvidia,cc35,ptxinfo"     \
      -DDO_CONCURRENT_FLAGS="-stdpar=gpu -Minfo=stdpar -gpu=cc35"     \
      -DFFLAGS="-O3 -DNO_INFORM"                                \
      -DLDFLAGS=""                                  \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      -DOUT_FREQ=1000 \
      ..

