#!/bin/bash

module load nvhpc

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_FC=nvfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                                                  \
      -DFFLAGS="-O3 -Mvect -Mextend -DNO_INFORM -I/opt/parallel-netcdf-1.12.0_nvhpc/include"           \
      -DLDFLAGS="-L/opt/parallel-netcdf-1.12.0_nvhpc/lib -lpnetcdf"                                    \
      -DOPENACC_FLAGS:STRING="-acc -gpu=cc50,fastmath,loadcache:L1,ptxinfo -Minfo=accel"               \
      -DOPENMP_FLAGS="-Minfo=mp"                                                                       \
      -DDO_CONCURRENT_FLAGS:STRING="-stdpar=gpu -Minfo=stdpar -gpu=cc50,fastmath,loadcache:L1,ptxinfo" \
      -DNX=200                                                                                         \
      -DNZ=100                                                                                         \
      -DSIM_TIME=1000                                                                                  \
      -DOUT_FREQ=2000                                                                                  \
      ..
