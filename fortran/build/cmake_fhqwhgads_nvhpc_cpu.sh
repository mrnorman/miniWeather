#!/bin/bash

source /usr/share/modules/init/bash
module purge
module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_FC=nvfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=/opt/parallel-netcdf-1.12.0_nvhpc   \
      -DFFLAGS="-O3 -Mvect -DNO_INFORM"                                \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      -DOUT_FREQ=2000 \
      ..

