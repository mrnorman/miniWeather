#!/bin/bash

source /usr/share/modules/init/bash
module purge

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_CXX=/opt/intel/oneapi/compiler/2022.0.2/linux/bin/icpx
export OMPI_CC=/opt/intel/oneapi/compiler/2022.0.2/linux/bin-llvm/clang
export OMPI_FC=/opt/intel/oneapi/compiler/2022.0.2/linux/bin/intel64/ifort

export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++         \
      -DCMAKE_C_COMPILER=mpicc            \
      -DCMAKE_Fortran_COMPILER=mpif90     \
      -DPNETCDF_PATH=${PNETCDF_PATH}      \
      -DYAKL_CXX_FLAGS="-Ofast -std=c++11 -DNO_INFORM"   \
      -DNX=200                            \
      -DNZ=100                            \
      -DSIM_TIME=1000                     \
      -DOUT_FREQ=2000 \
      ..

