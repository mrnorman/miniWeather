#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load nvhpc

export TEST_MPI_COMMAND="mpirun -n 1"

export OMPI_FC=nvfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DFFLAGS="-fastsse -O4 -Mfpapprox -Mfprelaxed -march=native -mtune=native -Mextend -DNO_INFORM -I/opt/parallel-netcdf-1.12.0_nvhpc/include"           \
      -DLDFLAGS="-L/opt/parallel-netcdf-1.12.0_nvhpc/lib -lpnetcdf"                                    \
      -DOPENMP_FLAGS="-mp -Minfo=mp"                                                                   \
      -DOPENACC_FLAGS:STRING="-acc -gpu=cc86,fastmath,loadcache:L1,pinned,unroll,fma,ptxinfo -Minfo=accel"               \
      -DOPENMP45_FLAGS:STRING="-Minfo=mp -mp=gpu -gpu=cc86,fastmath,loadcache:L1,pinned,unroll,fma,ptxinfo"              \
      -DDO_CONCURRENT_FLAGS:STRING="-stdpar=gpu -Minfo=stdpar -gpu=cc86,fastmath,loadcache:L2,unroll,fma,ptxinfo" \
      -DNX=256  \
      -DNZ=128  \
      -DSIM_TIME=250  \
      -DOUT_FREQ=2000 \
      ..

