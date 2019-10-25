
##########################################################
## EDIT THESE PARAMETERS
##########################################################
CXX := mpic++
CXXFLAGS := -O3 # -DARRAY_DEBUG
INCLUDE := -I${PNETCDF_PATH}/include
LDFLAGS := -L${PNETCDF_PATH}/lib -lpnetcdf #-L${LAPACK_PATH}/lib
# Valid DEVICE options: CPU, CUDA
DEVICE := CPU
# Valid GPU_ARCH options: https://github.com/kokkos/kokkos/wiki/Compiling#table-43-architecture-variables
GPU_ARCH := "Maxwell50"
MANAGED_MEM := yes
##########################################################
## END EDITING SECTION
##########################################################

KOKKOS_PATH = $(shell pwd)/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}

default: main

# KOKKOS_DEBUG = "yes"

ifeq ($(DEVICE),CPU)
  KOKKOS_DEVICES = "Serial"
endif
ifeq ($(DEVICE),CUDA)
  KOKKOS_DEVICES = "Cuda"
  KOKKOS_ARCH = $(GPU_ARCH)
  KOKKOS_CUDA_OPTIONS += "enable_lambda"
  CXXFLAGS += "-D__USE_CUDA__"
  ifeq ($(MANAGED_MEM),yes)
    CXXFLAGS += "-D__MANAGED__"
  endif
endif

include ${KOKKOS_PATH}/Makefile.kokkos

main: serial mpi

serial: miniWeather_serial.cpp *.h $(KOKKOS_CPP_DEPENDS) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(INCLUDE) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(EXTRA_PATH) $(EXTRA_INC) miniWeather_serial.cpp $(KOKKOS_LIBS) $(LIB) -o miniWeather_serial $(LDFLAGS)

mpi: miniWeather_mpi.cpp *.h $(KOKKOS_CPP_DEPENDS) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(INCLUDE) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(EXTRA_PATH) $(EXTRA_INC) miniWeather_mpi.cpp $(KOKKOS_LIBS) $(LIB) -o miniWeather_mpi $(LDFLAGS)

parallelfor: miniWeather_mpi.cpp *.h $(KOKKOS_CPP_DEPENDS) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(INCLUDE) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(EXTRA_PATH) $(EXTRA_INC) miniWeather_mpi_parallelfor.cpp $(KOKKOS_LIBS) $(LIB) -o miniWeather_mpi_parallelfor $(LDFLAGS)

clean:
	rm -f *.gch *.o *.dat miniWeather_mpi miniWeather_serial miniWeather_mpi_parallelfor KokkosCore_config.h KokkosCore_config.tmp libkokkos.a

