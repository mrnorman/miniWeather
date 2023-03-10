#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load nvhpc/23.1

export TEST_MPI_COMMAND="mpirun -n 1"

unset OMPI_CXX

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                                                                      \
      -DCXXFLAGS="-O3 -Mvect -DNO_INFORM -std=c++11 -I/opt/parallel-netcdf-1.12.0_nvhpc/include"       \
      -DLDFLAGS="-L/opt/parallel-netcdf-1.12.0_nvhpc/lib -lpnetcdf"                                    \
      -DOPENMP_FLAGS="-Minfo=mp -mp"                                                                   \
      -DOPENACC_FLAGS:STRING="-acc -gpu=cc86,fastmath,loadcache:L1,ptxinfo -Minfo=accel"               \
      -DOPENMP45_FLAGS:STRING="-Minfo=mp -mp=gpu -gpu=cc86,fastmath,loadcache:L1,ptxinfo"              \
      -DNX=200                                                                                         \
      -DNZ=100                                                                                         \
      -DSIM_TIME=1000                                                                                  \
      -DOUT_FREQ=1000                                                                                  \
      ..

