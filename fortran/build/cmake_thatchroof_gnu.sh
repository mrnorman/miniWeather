#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load gcc-12.1.0-gcc-11.1.0-g2ai6t2

export TEST_MPI_COMMAND="mpirun -n 1"

export OMPI_FC=gfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                     \
      -DOPENMP_FLAGS="-fopenmp"                                           \
      -DOPENACC_FLAGS="-fopenacc -ffast-math -foffload=nvptx-none=\"-lm -O3 -ffast-math -DSINGLE_PREC -march=sm_80 -moptimize\" -fopenacc-dim=16384:1:128 -DSINGLE_PREC -fopt-info-omp" \
      -DOPENMP45_FLAGS="-fopenmp -ffast-math -foffload=nvptx-none=\"-lm -O3 -latomic -ffast-math -DSINGLE_PREC -march=sm_80 -moptimize\" -DSINGLE_PREC -fopt-info-omp"                  \
      -DFFLAGS="-O3 -march=native -mtune=native -ffree-line-length-none -DNO_INFORM -DSINGLE_PREC -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"                   \
      -DNX=2048                                                           \
      -DNZ=1024                                                           \
      -DSIM_TIME=10                                                     \
      -DOUT_FREQ=20                                                     \
      ..

#      -DOPENACC_FLAGS="-fopenacc -fopenacc-dim=:1:128"                                         \
