SIM_TIME := 100.0
OUT_FREQ := 100.0
NX := 100
NZ := 50

DATA_SPEC_COLLISION       := 1
DATA_SPEC_THERMAL         := 2
DATA_SPEC_MOUNTAIN        := 3
DATA_SPEC_TURBULENCE      := 4
DATA_SPEC_DENSITY_CURRENT := 5
DATA_SPEC_INJECTION       := 6

DATA_SPEC := ${DATA_SPEC_COLLISION}
#DATA_SPEC := ${DATA_SPEC_THERMAL}

ACCEL_TYPE := fortran_openacc
ACCOUNT := cli133
WORK_DIR := ${PROJWORK}/${ACCOUNT}/${USER}/juliawork
DEBUG_DIR := ${WORK_DIR}
OUT_FILE  := ${WORK_DIR}/output.nc

JULIA := julia --project=.

SRCDIR :=../..
JLSRC := ${SRCDIR}/cpu/miniWeather_mpi.jl

ARGS := -s ${SIM_TIME} -x ${NX} -z ${NZ} -f ${OUT_FREQ} \
		-d ${DATA_SPEC} -o ${OUT_FILE}

CC := cc
CXX := CC
FC := ftn

INCLUDES := -I${OLCF_PARALLEL_NETCDF_ROOT}/include
LIBS := -L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf
MACROS := -D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}

F_FLAGS := ${INCLUDES} ${LIBS} ${MACROS}

cpu:
	${JULIA} --color=yes -O -- ${JLSRC} ${ARGS}

clean:
	rm -rf *.so core *.mod .jaitmp *.o *.s *.exe output.nc
