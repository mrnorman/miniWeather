#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu craype-accel-amd-gfx90a rocm
module use /sw/crusher/ums/compilers/modulefiles
module load gcc/13.2.1-dev-latest

export TEST_MPI_COMMAND="srun -n 1 --gpus-per-task 1 -c 1"
unset MPICH_GPU_SUPPORT_ENABLED

export MYINCLUDE="-I/lustre/orion/world-shared/stf010/mpich-gcc-include"
export MYLIBS="-L/opt/cray/pe/mpich/8.1.23/ofi/gnu/9.1/lib -L/opt/cray/pe/libsci/22.12.1.1/GNU/9.1/x86_64/lib -L/opt/cray/pe/dsmml/0.2.2/dsmml//lib -L/opt/cray/pe/pmi/6.1.8/lib -L/opt/cray/xpmem/2.6.2-2.5_2.22__gd067c3f.shasta/lib64 -Wl,--as-needed,-lsci_gnu_82_mpi,--no-as-needed -Wl,--as-needed,-lsci_gnu_82,--no-as-needed -ldl -Wl,--as-needed,-lmpifort_gnu_91,--no-as-needed -Wl,--as-needed,-lmpi_gnu_91,--no-as-needed -Wl,--as-needed,-ldsmml,--no-as-needed -Wl,--as-needed,-lpmi,--no-as-needed -Wl,--as-needed,-lpmi2,--no-as-needed -lxpmem -Wl,--as-needed,-lgfortran,-lquadmath,--no-as-needed -Wl,--as-needed,-lpthread,--no-as-needed -Wl,--disable-new-dtags"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=gfortran                                                     \
      -DFFLAGS="$MYINCLUDE -Ofast -march=native -DNO_INFORM -ffree-line-length-none -DNO_OUTPUT"         \
      -DLDFLAGS="$MYLIBS"                                      \
      -DOPENMP_FLAGS="-Ofast -march=native -fopenmp -DNO_INFORM -ffree-line-length-none -DNO_OUTPUT"                            \
      -DOPENACC_FLAGS="-Ofast -fopenacc -foffload=\"-lm -Ofast -march=gfx90a\" -DNO_INFORM -ffree-line-length-none -DNO_OUTPUT" \
      -DOPENMP45_FLAGS="-Ofast -fopenmp -foffload=\"-lm -Ofast -march=gfx90a\" -DNO_INFORM -ffree-line-length-none -DNO_OUTPUT" \
      -DNX=2048                                                                             \
      -DNZ=1024                                                                             \
      -DDATA_SPEC="DATA_SPEC_THERMAL"                                                       \
      -DSIM_TIME=100                                                                        \
      -DOUT_FREQ=-1                                                                         \
      ..


