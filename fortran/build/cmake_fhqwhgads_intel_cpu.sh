#!/bin/bash

source /usr/share/modules/init/bash
module purge
module use /opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga/modulefiles
module use /opt/intel/oneapi/compiler/2022.0.2/modulefiles
module use /opt/intel/oneapi/debugger/2021.5.0/modulefiles
module use /opt/intel/oneapi/dpcpp-ct/2022.0.0/modulefiles
module use /opt/intel/oneapi/mkl/2022.0.2/modulefiles
module use /opt/intel/oneapi/vpl/2022.0.0/modulefiles
module use /opt/intel/oneapi/vtune/2022.1.0/modulefiles
module use /opt/intel/oneapi/dal/2021.5.3/modulefiles
module use /opt/intel/oneapi/advisor/2022.0.0/modulefiles
module use /opt/intel/oneapi/ccl/2021.5.1/modulefiles
module use /opt/intel/oneapi/dpl/2021.6.0/modulefiles
module use /opt/intel/oneapi/dnnl/2022.0.2/modulefiles
module use /opt/intel/oneapi/ipp/2021.5.2/modulefiles
module use /opt/intel/oneapi/tbb/2021.5.1/modulefiles
module use /opt/intel/oneapi/dev-utilities/2021.5.2/modulefiles
module use /opt/intel/oneapi/ippcp/2021.5.1/modulefiles
module use /opt/intel/oneapi/mpi/2021.5.1/modulefiles
module load icc mpi

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_FC=/opt/intel/oneapi/compiler/2022.0.2/linux/bin/intel64/ifort

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=/opt/intel/oneapi/mpi/2021.5.1/bin/mpif90               \
      -DPNETCDF_PATH=${PNETCDF_PATH}   \
      -DFFLAGS="-Ofast -march=native -mtune=native -DNO_INFORM -DNO_OUTPUT"                                \
      -DNX=256 \
      -DNZ=128 \
      -DSIM_TIME=250  \
      -DOUT_FREQ=2000 \
      ..

