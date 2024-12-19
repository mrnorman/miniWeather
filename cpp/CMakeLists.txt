cmake_minimum_required(VERSION 3.16)
project(miniWeather CXX)
enable_testing()

include(cmake/utils.cmake)

option(MW_ENABLE_SERIAL "Whether to build serial variant" ON)
option(MW_ENABLE_MPI "Whether to build MPI variant" OFF)
option(MW_ENABLE_MPI_OPENMP "Whether to build MPI/OPENMP variant" OFF)
option(MW_ENABLE_MPI_OPENMP45 "Whether to build MPI/OPENMP45 variant" OFF)
option(MW_ENABLE_MPI_OPENACC "Whether to build MPI/OPENACC variant" OFF)
option(MW_ENABLE_ALL "Whether to build all variant" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

if (MW_ENABLE_SERIAL OR MW_ENABLE_ALL)
  set(VERSION SERIAL)
  list(APPEND VARIANTS ${VERSION})
endif()
if (MW_ENABLE_MPI OR MW_ENABLE_ALL)
  set(VERSION MPI)
  list(APPEND VARIANTS ${VERSION})
endif()
if (MW_ENABLE_MPI_OPENMP OR MW_ENABLE_ALL)
 set(VERSION MPI_OPENMP)
  list(APPEND VARIANTS ${VERSION})
endif()
if (MW_ENABLE_MPI_OPENMP45 OR MW_ENABLE_ALL)
 set(VERSION MPI_OPENMP45)
  list(APPEND VARIANTS ${VERSION})
endif()
if (MW_ENABLE_MPI_OPENACC OR MW_ENABLE_ALL)
 set(VERSION MPI_OPENACC)
  list(APPEND VARIANTS ${VERSION})
endif()

message(STATUS "Enabled variant(s): ${VARIANTS}")

list(LENGTH VARIANTS N_VARIANTS)
if (${N_VARIANTS} EQUAL "0")
  message(FATAL_ERROR "Must enable a valid implementation variant. ${N_VARIANTS} given.")
endif()

find_package(MPI REQUIRED)
add_library(MPI INTERFACE)
list(APPEND MPI_CXX_LINK_FLAGS ${MPI_CXX_LIBRARIES})
set_target_properties(MPI PROPERTIES
  INTERFACE_COMPILE_OPTIONS "${MPI_CXX_COMPILE_FLAGS}"
  INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}"
  INTERFACE_LINK_LIBRARIES "${MPI_CXX_LINK_FLAGS}"
)
list(APPEND PUBLIC_DEPS MPI)

find_package(PnetCDF REQUIRED)
list(APPEND PUBLIC_DEPS PnetCDF)

############################################################
## Set Parameters
############################################################
if ("${NX}" STREQUAL "")
  SET(NX 100)
endif()
if ("${NZ}" STREQUAL "")
  SET(NZ 50)
endif()
if ("${SIM_TIME}" STREQUAL "")
  SET(SIM_TIME 1000)
endif()
if ("${OUT_FREQ}" STREQUAL "")
  SET(OUT_FREQ 10)
endif()
if ("${DATA_SPEC}" STREQUAL "")
  SET(DATA_SPEC DATA_SPEC_THERMAL)
endif()

SET(EXE_DEFS "-D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}")
SET(TEST_DEFS "-D_NX=100 -D_NZ=50 -D_SIM_TIME=400 -D_OUT_FREQ=400 -D_DATA_SPEC=DATA_SPEC_THERMAL")

############################################################
## Gen Targets
############################################################
foreach(VARIANT ${VARIANTS})
  string(TOLOWER ${VARIANT} VARIANT_LOWER)
  set(SUFFIX "_${VARIANT_LOWER}")
  set(SRC_NAME miniWeather${SUFFIX}.cpp)
  set(BIN_NAME "${CMAKE_PROJECT_NAME}${SUFFIX}")
  set(BIN_TEST_NAME "${BIN_NAME}_test")

  #Default
  add_executable(${BIN_NAME} ${SRC_NAME})
  set_target_properties(${BIN_NAME} PROPERTIES COMPILE_FLAGS ${EXE_DEFS})
  target_link_libraries(${BIN_NAME} PUBLIC ${PUBLIC_DEPS})

  #Test
  add_executable(${BIN_TEST_NAME} ${SRC_NAME})
  set_target_properties(${BIN_TEST_NAME} PROPERTIES COMPILE_FLAGS ${TEST_DEFS})
  target_link_libraries(${BIN_TEST_NAME} PUBLIC ${PUBLIC_DEPS})

  #check_output
  add_test(NAME ${BIN_TEST_NAME} COMMAND ./scripts/check_output.sh ./${BIN_TEST_NAME} 1e-13 4.5e-5 )
endforeach()

