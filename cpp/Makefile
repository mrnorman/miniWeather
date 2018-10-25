
##########################################################
## EDIT THESE PARAMETERS
##########################################################
CC := mpiCC
CFLAGS := -O3
INCLUDE := -I${PNETCDF_PATH}/include
LDFLAGS := -L${PNETCDF_PATH}/lib -lpnetcdf -L${LAPACK_PATH}/lib
OMPFLAGS := -mp
ACCFLAGS := -ta=tesla,pinned,cc50,cuda9.0,ptxinfo -Minfo=accel
##########################################################
## END EDITING SECTION
##########################################################

all: serial #mpi openmp openacc

serial:
	${CC} ${INCLUDE} ${CFLAGS} ${ACCFLAGS} -o miniWeather_serial miniWeather_serial.cpp ${LDFLAGS}

mpi:
	${CC} ${INCLUDE} ${CFLAGS} -o miniWeather_mpi miniWeather_mpi.cpp ${LDFLAGS}

openmp:
	${CC} ${INCLUDE} ${CFLAGS} ${OMPFLAGS} -o miniWeather_mpi_openmp miniWeather_mpi_openmp.cpp ${LDFLAGS}

openacc:
	${CC} ${INCLUDE} ${CFLAGS} ${ACCFLAGS} -o miniWeather_mpi_openacc miniWeather_mpi_openacc.cpp ${LDFLAGS}

clean:
	rm -f *.o *.mod miniWeather_serial miniWeather_mpi miniWeather_mpi_openmp miniWeather_mpi_openacc
