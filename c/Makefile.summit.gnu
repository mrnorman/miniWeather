##########################################################
## EDIT THESE PARAMETERS
##########################################################
CC := mpic++
CFLAGS := -O3
INCLUDE := -I${OLCF_PARALLEL_NETCDF_ROOT}/include
LDFLAGS := -L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf
OMPFLAGS := -fopenmp
OMP45FLAGS := -fopenmp
ACCFLAGS := -fopenacc
##########################################################
## END EDITING SECTION
##########################################################

all: serial mpi openmp openacc

serial:
	${CC} ${INCLUDE} ${CFLAGS} -o miniWeather_serial miniWeather_serial.cpp ${LDFLAGS}

mpi:
	${CC} ${INCLUDE} ${CFLAGS} -o miniWeather_mpi miniWeather_mpi.cpp ${LDFLAGS}

openmp:
	${CC} ${INCLUDE} ${CFLAGS} ${OMPFLAGS} -o miniWeather_mpi_openmp miniWeather_mpi_openmp.cpp ${LDFLAGS}

openacc:
	${CC} ${INCLUDE} ${CFLAGS} ${ACCFLAGS} -o miniWeather_mpi_openacc miniWeather_mpi_openacc.cpp ${LDFLAGS}

clean:
	rm -f *.o *.mod miniWeather_serial miniWeather_mpi miniWeather_mpi_openmp miniWeather_mpi_openacc

