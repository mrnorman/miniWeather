
##########################################################
## EDIT THESE PARAMETERS
##########################################################
CXX := mpic++
CXXFLAGS := -O3 # -DARRAY_DEBUG
INCLUDE := -I${PNETCDF_PATH}/include
LDFLAGS := -L${PNETCDF_PATH}/lib -lpnetcdf
DEVICE := HIP       # Valid DEVICE options: CPU, CUDA, HIP
CUDA_ARCH := sm_50  # For more info: nvcc --help | grep -A 30 "gpu-architecture <arch>"
MANAGED_MEM := no   # [yes | no]
##########################################################
## END EDITING SECTION
##########################################################

default: main

ifeq ($(DEVICE),CUDA)
  CXX := nvcc -x cu -ccbin mpic++ -arch $(CUDA_ARCH) -I ./cub --use_fast_math --expt-extended-lambda
  CXXFLAGS += -D__USE_CUDA__
  ifeq ($(MANAGED_MEM),yes)
    CXXFLAGS += -D__MANAGED__
  endif
endif
ifeq ($(DEVICE),HIP)
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
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c $< 

clean:
	rm -f *.gch *.o *.dat miniWeather_mpi miniWeather_serial miniWeather_mpi_parallelfor

