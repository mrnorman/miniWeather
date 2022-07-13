#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps nvhpc/21.11 cuda parallel-netcdf cmake

export TEST_MPI_COMMAND="mpirun -n 1"

export OMPI_FC=nvfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DFFLAGS="-DSINGLE_PREC -fastsse -O4 -Mfpapprox -Mfprelaxed -Mextend -DNO_INFORM -I${OLCF_PARALLEL_NETCDF_ROOT}/include"           \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                                    \
      -DOPENMP_FLAGS="-mp -Minfo=mp"                                                                   \
      -DOPENACC_FLAGS:STRING="-acc -gpu=cc70,fastmath,loadcache:L1,pinned,unroll,fma,ptxinfo -Minfo=accel"               \
      -DOPENMP45_FLAGS:STRING="-Minfo=mp -mp=gpu -gpu=cc70,fastmath,loadcache:L1,pinned,unroll,fma,ptxinfo"              \
      -DDO_CONCURRENT_FLAGS:STRING="-stdpar=gpu -Minfo=stdpar -gpu=cc70,fastmath,loadcache:L1,unroll,fma,ptxinfo" \
      -DNX=2048 \
      -DNZ=1024 \
      -DSIM_TIME=10   \
      -DOUT_FREQ=2000 \
      ..

