#!/bin/bash

# BEFORE RUNNING THIS SCRIPT ON LYRA:

# module load rocm hip openmpi cmake
# export OMPI_CXX=hipcc

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpic++                   \
      -DPNETCDF_PATH=/ccs/home/imn/parallel-netcdf-1.11.2_clang \
      -DYAKL_HIPCUB_HOME=`pwd`/../hipCUB            \
      -DYAKL_ROCPRIM_HOME=`pwd`/../rocPRIM          \
      -DCXXFLAGS="-O3 -std=c++11"                   \
      -DARCH="HIP"                                  \
      -DHIP_FLAGS=""                                \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      ..

