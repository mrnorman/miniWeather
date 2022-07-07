#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load icc mpi

export TEST_MPI_COMMAND="mpirun -n 1"

export I_MPI_FC=ifort
export I_MPI_F90=ifort

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DFFLAGS="-Ofast -march=native -mtune=native -DNO_INFORM -DSINGLE_PREC -I/opt/parallel-netcdf-1.12.0_intel/include"           \
      -DLDFLAGS="-L/opt/parallel-netcdf-1.12.0_intel/lib -lpnetcdf"                                    \
      -DOPENMP_FLAGS="-qopenmp"                                                                   \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      -DOUT_FREQ=1000 \
      ..

