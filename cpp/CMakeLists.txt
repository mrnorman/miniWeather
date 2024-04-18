cmake_minimum_required(VERSION 3.8)
project(miniWeather CXX)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

enable_testing()


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

if ( ("${YAKL_CXX_FLAGS}"    MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_CUDA_FLAGS}"   MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_HIP_FLAGS}"    MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_OPENMP_FLAGS}" MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_SYCL_FLAGS}"   MATCHES ".*SINGLE_PREC.*") )
  message(STATUS "Using single precision")
else()
  message(STATUS "Using double precision")
endif()

############################################################
## Append CXXFLAGS
############################################################
SET(CMAKE_CXX_FLAGS "${CXXFLAGS}")


############################################################
## Add YAKL 
############################################################
add_subdirectory(YAKL)
if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
  set_property(TARGET yakl PROPERTY CUDA_ARCHITECTURES OFF)
endif()
include_directories(YAKL)



############################################################
## Compile the serial version
############################################################
add_executable(serial miniWeather_serial.cpp)
set_target_properties(serial PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(serial_test miniWeather_serial.cpp)
set_target_properties(serial_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

target_link_libraries(serial      "${LDFLAGS}")
target_link_libraries(serial_test "${LDFLAGS}")

add_test(NAME SERIAL_TEST COMMAND ./check_output.sh ./serial_test 1e-9 4.5e-5 )

############################################################
## Compile the serial for_each version
############################################################
if (NOT ("${STDPAR_FLAGS}" STREQUAL "") )
  add_executable(serial_for_each miniWeather_serial_for_each.cpp)
  set_target_properties(serial_for_each PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${STDPAR_FLAGS}")
  
  add_executable(serial_for_each_test miniWeather_serial_for_each.cpp)
  set_target_properties(serial_for_each_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${STDPAR_FLAGS}")
  
  if (NOT ("${LDFLAGS}" STREQUAL "") )
    target_link_libraries(serial_for_each      "${LDFLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(serial_for_each_test "${LDFLAGS} ${STDPAR_FLAGS}")
  endif()
  if (NOT ("${SERIAL_LINK_FLAGS}" STREQUAL "") )
    target_link_libraries(serial_for_each      "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(serial_for_each_test "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
  endif()
  
  add_test(NAME SERIAL_FOR_EACH_TEST COMMAND ./check_output.sh ./serial_for_each_test 1e-13 4.5e-5 ) 
endif()

############################################################
## Compile the serial for_each_n version
############################################################
if (NOT ("${STDPAR_FLAGS}" STREQUAL "") )
  add_executable(serial_for_each_n miniWeather_serial_for_each_n.cpp)
  set_target_properties(serial_for_each_n PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${STDPAR_FLAGS}")
  
  add_executable(serial_for_each_n_test miniWeather_serial_for_each_n.cpp)
  set_target_properties(serial_for_each_n_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${STDPAR_FLAGS}")
  
  if (NOT ("${LDFLAGS}" STREQUAL "") )
    target_link_libraries(serial_for_each_n      "${LDFLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(serial_for_each_n_test "${LDFLAGS} ${STDPAR_FLAGS}")
  endif()
  if (NOT ("${SERIAL_LINK_FLAGS}" STREQUAL "") )
    target_link_libraries(serial_for_each_n      "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(serial_for_each_n_test "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
  endif()
  
  add_test(NAME SERIAL_FOR_EACH_N_TEST COMMAND ./check_output.sh ./serial_for_each_n_test 1e-13 4.5e-5 ) 
endif()



############################################################
## Compile the MPI version
############################################################
add_executable(mpi miniWeather_mpi.cpp)
set_target_properties(mpi PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(mpi_test miniWeather_mpi.cpp)
set_target_properties(mpi_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

target_link_libraries(mpi      "${LDFLAGS}")
target_link_libraries(mpi_test "${LDFLAGS}")

add_test(NAME MPI_TEST COMMAND ./check_output.sh ./mpi_test 1e-9 4.5e-5 ) 

############################################################
## Compile the mpi for_each version
############################################################
if (NOT ("${STDPAR_FLAGS}" STREQUAL "") )
  add_executable(mpi_for_each miniWeather_mpi_for_each.cpp)
  set_target_properties(mpi_for_each PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${STDPAR_FLAGS}")
  
  add_executable(mpi_for_each_test miniWeather_mpi_for_each.cpp)
  set_target_properties(mpi_for_each_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${STDPAR_FLAGS}")
  
  if (NOT ("${LDFLAGS}" STREQUAL "") )
    target_link_libraries(mpi_for_each      "${LDFLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(mpi_for_each_test "${LDFLAGS} ${STDPAR_FLAGS}")
  endif()
  if (NOT ("${SERIAL_LINK_FLAGS}" STREQUAL "") )
    target_link_libraries(mpi_for_each      "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(mpi_for_each_test "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
  endif()
  
  add_test(NAME MPI_FOR_EACH_TEST COMMAND ./check_output.sh ./mpi_for_each_test 1e-13 4.5e-5 ) 
endif()

############################################################
## Compile the mpi for_each_n version
############################################################
if (NOT ("${STDPAR_FLAGS}" STREQUAL "") )
  add_executable(mpi_for_each_n miniWeather_mpi_for_each_n.cpp)
  set_target_properties(mpi_for_each_n PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${STDPAR_FLAGS}")
  
  add_executable(mpi_for_each_n_test miniWeather_mpi_for_each_n.cpp)
  set_target_properties(mpi_for_each_n_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${STDPAR_FLAGS}")
  
  if (NOT ("${LDFLAGS}" STREQUAL "") )
    target_link_libraries(mpi_for_each_n      "${LDFLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(mpi_for_each_n_test "${LDFLAGS} ${STDPAR_FLAGS}")
  endif()
  if (NOT ("${SERIAL_LINK_FLAGS}" STREQUAL "") )
    target_link_libraries(mpi_for_each_n      "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
    target_link_libraries(mpi_for_each_n_test "${SERIAL_LINK_FLAGS} ${STDPAR_FLAGS}")
  endif()
  
  add_test(NAME MPI_FOR_EACH_N_TEST COMMAND ./check_output.sh ./mpi_for_each_n_test 1e-13 4.5e-5 ) 
endif()


############################################################
## YAKL parallel_for Version
############################################################
add_executable(parallelfor miniWeather_mpi_parallelfor.cpp)
set_target_properties(parallelfor PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(parallelfor_test miniWeather_mpi_parallelfor.cpp)
set_target_properties(parallelfor_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

target_link_libraries(parallelfor      "${LDFLAGS}")
target_link_libraries(parallelfor_test "${LDFLAGS}")

add_test(NAME YAKL_TEST COMMAND ./check_output.sh ./parallelfor_test 1e-9 4.5e-5 )


############################################################
## YAKL parallel_for_simd_x Version
############################################################
add_executable(parallelfor_simd_x experimental/miniWeather_mpi_parallelfor_simd_x.cpp)
set_target_properties(parallelfor_simd_x PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(parallelfor_simd_x_test experimental/miniWeather_mpi_parallelfor_simd_x.cpp)
set_target_properties(parallelfor_simd_x_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

target_link_libraries(parallelfor_simd_x      "${LDFLAGS}")
target_link_libraries(parallelfor_simd_x_test "${LDFLAGS}")

add_test(NAME YAKL_SIMD_X_TEST COMMAND ./check_output.sh ./parallelfor_simd_x_test 1e-9 4.5e-5 )


include(YAKL/yakl_utils.cmake)
yakl_process_target(serial)
yakl_process_target(serial_test)
yakl_process_target(mpi)
yakl_process_target(mpi_test)
yakl_process_target(parallelfor)
yakl_process_target(parallelfor_test)
yakl_process_target(parallelfor_simd_x)
yakl_process_target(parallelfor_simd_x_test)


if ("${YAKL_ARCH}" STREQUAL "CUDA")
  set_target_properties(serial serial_test mpi mpi_test parallelfor parallelfor_test parallelfor_simd_x parallelfor_simd_x_test  PROPERTIES LINKER_LANGUAGE CXX)
  if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_target_properties(serial serial_test mpi mpi_test parallelfor parallelfor_test parallelfor_simd_x parallelfor_simd_x_test  PROPERTIES CUDA_ARCHITECTURES OFF)
  endif()
endif()



