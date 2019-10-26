
##########################################################
## EDIT THESE PARAMETERS
##########################################################
CXX := mpic++
CXXFLAGS := -O3 # -DARRAY_DEBUG
INCLUDE := -I${PNETCDF_PATH}/include
LDFLAGS := -L${PNETCDF_PATH}/lib -lpnetcdf #-L${LAPACK_PATH}/lib
# Valid DEVICE options: CPU, CUDA, HIP
DEVICE := CPU
CUDA_ARCH := sm_50
MANAGED_MEM := no
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

main: serial mpi parallelfor

serial: miniWeather_serial.cpp *.h 
	$(CXX) $(INCLUDE) $(CXXFLAGS) $(EXTRA_PATH) $(EXTRA_INC) miniWeather_serial.cpp $(LIB) -o miniWeather_serial $(LDFLAGS)

mpi: miniWeather_mpi.cpp *.h
	$(CXX) $(INCLUDE) $(CXXFLAGS) $(EXTRA_PATH) $(EXTRA_INC) miniWeather_mpi.cpp $(LIB) -o miniWeather_mpi $(LDFLAGS)

parallelfor: miniWeather_mpi.cpp *.h
	$(CXX) $(INCLUDE) $(CXXFLAGS) $(EXTRA_PATH) $(EXTRA_INC) miniWeather_mpi_parallelfor.cpp $(LIB) -o miniWeather_mpi_parallelfor $(LDFLAGS)

clean:
	rm -f *.gch *.o *.dat miniWeather_mpi miniWeather_serial miniWeather_mpi_parallelfor

