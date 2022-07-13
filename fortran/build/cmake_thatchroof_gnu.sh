#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load gcc-12.1.0-gcc-11.1.0-g2ai6t2

export TEST_MPI_COMMAND="mpirun -n 1"

export OMPI_FC=gfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                     \
      -DOPENMP_FLAGS="-fopenmp"                                           \
      -DOPENACC_FLAGS="-fopenacc -foffload=nvptx-none=\"-lm -Ofast -ffast-math -march=sm_80 -moptimize\" -fopenacc-dim=16384:1:128 -fopt-info-omp" \
      -DOPENMP45_FLAGS="-fopenmp -foffload=nvptx-none=\"-lm -Ofast -latomic -ffast-math -march=sm_80 -moptimize\" -fopt-info-omp"                  \
      -DFFLAGS="-Ofast -ffast-math -march=native -mtune=native -ffree-line-length-none -DNO_INFORM -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"                   \
      -DNX=256                                                            \
      -DNZ=128                                                            \
      -DSIM_TIME=250                                                    \
      -DOUT_FREQ=2000                                                   \
      ..

#      -DOPENACC_FLAGS="-fopenacc -fopenacc-dim=:1:128"                                         \
