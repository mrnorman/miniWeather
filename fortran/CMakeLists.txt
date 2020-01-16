cmake_minimum_required(VERSION 2.8)
project(miniWeather)

enable_language(Fortran)
enable_testing()

if ("${PNETCDF_PATH}" STREQUAL "")
  MESSAGE(FATAL_ERROR "ERROR: PNETCDF_PATH CMake variable not set."
                      "miniWeather requires a parallel-netcdf installation <https://trac.mcs.anl.gov/projects/parallel-netcdf>."
                      "Specify -DPNETCDF_PATH=/path/to/pnetcdf in your cmake configure line.\n")
endif()

############################################################
## Compile the serial version
############################################################
add_executable(miniWeather_serial miniWeather_serial.F90)
target_include_directories(miniWeather_serial PUBLIC ${PNETCDF_PATH}/include)
target_link_libraries(miniWeather_serial "-L${PNETCDF_PATH}/lib -lpnetcdf")

############################################################
## Compile the MPI version
############################################################
add_executable(miniWeather_mpi miniWeather_mpi.F90)
target_include_directories(miniWeather_mpi PUBLIC ${PNETCDF_PATH}/include)
target_link_libraries(miniWeather_mpi "-L${PNETCDF_PATH}/lib -lpnetcdf")

############################################################
## Compile the MPI + OpenMP version
############################################################
add_executable(miniWeather_mpi_openmp miniWeather_mpi_openmp.F90)
target_include_directories(miniWeather_mpi_openmp PUBLIC ${PNETCDF_PATH}/include)
target_link_libraries(miniWeather_mpi_openmp "-L${PNETCDF_PATH}/lib -lpnetcdf")


############################################################
## Check to see if the compiler is OpenACC-capable
############################################################
# If user didn't specify an OpenACC-capable compiler list, then create one
if ("${OPENACC_LIST}" STREQUAL "")
  SET(OPENACC_LIST GNU PGI)
  MESSAGE(STATUS "OpenACC-capable list is: ${OPENACC_LIST}")
endif()

# Check compiler against openacc-capable list
SET(DO_OPENACC FALSE)
list(FIND OPENACC_LIST ${CMAKE_Fortran_COMPILER_ID} _ind)
if( ${_ind} GREATER -1 )
  SET(DO_OPENACC TRUE)
  # GNU is only openacc-capable with 8.0 onward
  if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU" AND ${CMAKE_Fortran_COMPILER_VERSION} LESS 8)
    MESSAGE(WARNING "WARNING: GNU can only compile OpenACC code with versions >= 8.0"
                    "Not compiling OpenACC code.\n")
    SET(DO_OPENACC FALSE)
  endif()
  # If OPENACC_FLAGS isn't set, then do not compile the openacc code
  if ($"{OPENACC_FLAGS}" STREQUAL "")
    MESSAGE(WARNING "WARNING: OPENACC_FLAGS CMake variable must be set to compile OpenACC code.\n"
                    "add -DOPENACC_FLAGS=\"...\" to your cmake configure line to enable OpenACC compilation.\n")
    SET(DO_OPENACC FALSE)
  endif()
endif()


if (${DO_OPENACC})
  MESSAGE(STATUS "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} IS OpenACC-capable")
  ############################################################
  ## Compile the MPI + OpenACC version
  ############################################################
  add_executable(miniWeather_mpi_openacc miniWeather_mpi_openacc.F90)
  target_include_directories(miniWeather_mpi_openacc PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(miniWeather_mpi_openacc COMPILE_FLAGS "${OPENACC_FLAGS}")
  if ("${OPENACC_LINK_FLAGS}" STREQUAL "")
    MESSAGE(WARNING "WARNING: OPENACC_LINK_FLAGS CMake variable not specified."
                    "Using OPENACC_FLAGS in place of OPENACC_LINK_FLAGS in the link phase")
    SET(OPENACC_LINK_FLAGS ${OPENACC_FLAGS})
  endif()
  target_link_libraries(miniWeather_mpi_openacc "-L${PNETCDF_PATH}/lib -lpnetcdf ${OPENACC_LINK_FLAGS}")
else() 
  MESSAGE(STATUS "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} is NOT OpenACC-capable")
endif()


############################################################
## Check to see if the compiler is OpenMP-offload-capable
############################################################
# If user didn't specify an OpenMP4.5-capable compiler list, then create one
if ("${OPENMP45_LIST}" STREQUAL "")
  SET(OPENMP45_LIST XL GNU)
  MESSAGE(STATUS "OpenMP4.5-capable list is: ${OPENMP45_LIST}")
endif()

# Check compiler against openmp4.5-capable list
SET(DO_OPENMP45 FALSE)
list(FIND OPENMP45_LIST ${CMAKE_Fortran_COMPILER_ID} _ind)
if( ${_ind} GREATER -1 )
  SET(DO_OPENMP45 TRUE)
  # GNU is only openmp4.5-capable with 8.0 onward
  if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU" AND ${CMAKE_Fortran_COMPILER_VERSION} LESS 8)
    MESSAGE(WARNING "WARNING: GNU can only compile OpenMP4.5 code with versions >= 8.0"
                    "Not compiling OpenMP4.5 code.\n")
    SET(DO_OPENMP45 FALSE)
  endif()
  # If OPENMP45_FLAGS isn't set, then do not compile the openmp4.5 code
  if ($"{OPENMP45_FLAGS}" STREQUAL "")
    MESSAGE(WARNING "WARNING: OPENMP45_FLAGS CMake variable must be set to compile OpenMP4.5 code.\n"
                    "add -DOPENMP45_FLAGS=\"...\" to your cmake configure line to enable OpenMP4.5 compilation.\n")
    SET(DO_OPENMP45 FALSE)
  endif()
endif()

if (${DO_OPENMP45})
  MESSAGE(STATUS "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} IS OpenMP4.5-capable")
  ############################################################
  ## Compile the MPI + OpenMP4.5 version
  ############################################################
  add_executable(miniWeather_mpi_openmp45 miniWeather_mpi_openmp45.F90)
  target_include_directories(miniWeather_mpi_openmp45 PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(miniWeather_mpi_openmp45 COMPILE_FLAGS "${OPENMP45_FLAGS}")
  if ("${OPENMP45_LINK_FLAGS}" STREQUAL "")
    MESSAGE(WARNING "WARNING: OPENMP45_LINK_FLAGS CMake variable not specified."
                    "Using OPENMP45_FLAGS in place of OPENMP45_LINK_FLAGS in the link phase")
    SET(OPENMP45_LINK_FLAGS ${OPENMP45_FLAGS})
  endif()
  target_link_libraries(miniWeather_mpi_openmp45 "-L${PNETCDF_PATH}/lib -lpnetcdf ${OPENMP45_LINK_FLAGS}")
else() 
  MESSAGE(STATUS "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} is NOT OpenMP4.5-capable")
endif()

