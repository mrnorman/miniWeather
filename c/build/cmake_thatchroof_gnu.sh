#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load gcc-12.1.0-gcc-11.1.0-g2ai6t2

export TEST_MPI_COMMAND="mpirun -n 1"

export OMPI_CXX=g++

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                                         \
      -DCXXFLAGS="-O3 -march=native -std=c++11"                           \
      -DOPENMP_FLAGS="-fopenmp"                                           \
      -DOPENACC_FLAGS="-fopenacc -foffload=nvptx-none=\"-lm -Ofast -ffast-math -march=sm_80 -moptimize\" -fopenacc-dim=16384:1:128 -fopt-info-omp" \
      -DOPENMP45_FLAGS="-fopenmp -foffload=nvptx-none=\"-lm -Ofast -latomic -ffast-math -march=sm_80 -moptimize\" -fopt-info-omp"                  \
      -DLDFLAGS="-L/usr/lib/x86_64-linux-gnu -lpnetcdf"                   \
      -DNX=200                                                            \
      -DNZ=100                                                            \
      -DSIM_TIME=1000                                                     \
      ..
