
##########################################################
## EDIT THESE PARAMETERS
##########################################################
CXX := mpic++
CXXFLAGS := -O3 # -DARRAY_DEBUG
INCLUDE := -I${PNETCDF_PATH}/include
LDFLAGS := -L${PNETCDF_PATH}/lib -lpnetcdf #-L${LAPACK_PATH}/lib
##########################################################
## END EDITING SECTION
##########################################################

KOKKOS_PATH = $(shell pwd)/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}

default: main

KOKKOS_DEVICES = "Serial"
# KOKKOS_DEBUG = "yes"

include ${KOKKOS_PATH}/Makefile.kokkos

main: serial

serial: miniWeather_serial.cpp *.h $(KOKKOS_CPP_DEPENDS) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(INCLUDE) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(EXTRA_PATH) $(EXTRA_INC) miniWeather_serial.cpp $(KOKKOS_LIBS) $(LIB) -o miniWeather_serial $(LDFLAGS)

clean:
	rm -f *.gch *.o *.dat miniWeather_serial KokkosCore_config.h KokkosCore_config.tmp libkokkos.a

