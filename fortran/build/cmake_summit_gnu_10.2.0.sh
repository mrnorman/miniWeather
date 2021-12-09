#!/bin/bash

##############################################################################
## This requires gcc/8.1.1 on Summit right now
##############################################################################

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/10.2.0 cuda cmake

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-9.3.0/parallel-netcdf-1.8.1-xpruvxtexuss76fogjekuno3n4ipl3uq   \
      -DOPENACC_FLAGS="-fopenacc -foffload=\"-lm -O3\" " \
      -DOPENMP_FLAGS="-fopenmp"                     \
      -DOPENMP45_FLAGS="-fopenmp -foffload=\"-lm -O3\" " \
      -DFFLAGS="-O3"                                \
      -DLDFLAGS=""                                  \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      ..

