
##########################################################
## EDIT THESE PARAMETERS
##########################################################
FC := mpif90
FFLAGS := -O3 -ffree-line-length-none
INCLUDE := -I${OLCF_PARALLEL_NETCDF_ROOT}/include
LDFLAGS := -L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf
OMPFLAGS := -fopenmp
ACCFLAGS := -fopenacc
OMP45FLAGS := -fopenmp
##########################################################
## END EDITING SECTION
##########################################################

all: serial mpi openmp openacc

serial:
	${FC} ${INCLUDE} ${FFLAGS} -o miniWeather_serial miniWeather_serial.F90 ${LDFLAGS}

mpi:
	${FC} ${INCLUDE} ${FFLAGS} -o miniWeather_mpi miniWeather_mpi.F90 ${LDFLAGS}

openmp:
	${FC} ${INCLUDE} ${FFLAGS} ${OMPFLAGS} -o miniWeather_mpi_openmp miniWeather_mpi_openmp.F90 ${LDFLAGS}

openmp45:
	${FC} ${INCLUDE} ${FFLAGS} ${OMP45FLAGS} -o miniWeather_mpi_openmp45 miniWeather_mpi_openmp45.F90 ${LDFLAGS}

openacc:
	${FC} ${INCLUDE} ${FFLAGS} ${ACCFLAGS} -o miniWeather_mpi_openacc miniWeather_mpi_openacc.F90 ${LDFLAGS}

clean:
	rm -f *.o *.mod miniWeather_serial miniWeather_mpi miniWeather_mpi_openmp miniWeather_mpi_openacc
