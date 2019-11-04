
##########################################################
## EDIT THESE PARAMETERS
##########################################################
CXX := mpic++
CXXFLAGS := -O3 -DARRAY_DEBUG
INCLUDE := -I${PNETCDF_PATH}/include
LDFLAGS := -L${PNETCDF_PATH}/lib -lpnetcdf
# Valid DEVICE options: CPU, CUDA, HIP
DEVICE := CPU
# For more info on CUDA_ARCH: nvcc --help | grep -A 30 "gpu-architecture <arch>"
CUDA_ARCH := sm_35
# MANAGED_MEM := [yes | no]
MANAGED_MEM := no
##########################################################
## END EDITING SECTION
##########################################################

default: main

ifeq ($(DEVICE),CUDA)
  $(info Compiling for CUDA)
  CXX := nvcc -ccbin mpic++
  CXXFLAGS += -D__USE_CUDA__ -arch $(CUDA_ARCH) -I ./cub --use_fast_math --expt-extended-lambda
  CUDACXXFLAGS = -x cu
  ifeq ($(MANAGED_MEM),yes)
    $(info Using Managed Memory)
    CXXFLAGS += -D__MANAGED__
  endif
endif
ifeq ($(DEVICE),HIP)
  $(info Compiling for HIP)
  export OMPI_CXX=hipcc
  CXXFLAGS += -D__USE_HIP__
endif

main: serial mpi parallelfor

serial: miniWeather_serial.o YAKL.o
	$(CXX) $(CXXFLAGS) $(LIB) -o miniWeather_serial $^ $(LDFLAGS)

mpi: miniWeather_mpi.o YAKL.o
	$(CXX) $(CXXFLAGS) $(LIB) -o miniWeather_mpi $^ $(LDFLAGS)

parallelfor: miniWeather_mpi_parallelfor.o YAKL.o
	$(CXX) $(CXXFLAGS) $(LIB) -o miniWeather_mpi_parallelfor $^ $(LDFLAGS)

%.o:%.cpp *.h
	$(CXX) $(INCLUDE) $(CXXFLAGS) $(CUDACXXFLAGS) -c $< 

clean:
	rm -f *.gch *.o *.dat miniWeather_mpi miniWeather_serial miniWeather_mpi_parallelfor

