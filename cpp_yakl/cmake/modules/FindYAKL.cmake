find_library(yakl_fortran_lib_found yakl_fortran PATHS ${YAKL_ROOT}/lib NO_DEFAULT_PATHS)
find_path(yakl_c_headers_found YAKL.h PATHS ${YAKL_ROOT}/include NO_DEFAULT_PATHS)

if (yakl_fortran_lib_found AND yakl_c_headers_found)
  add_library(YAKL INTERFACE)
  set_target_properties(YAKL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${yakl_c_headers_found}
  )
else()
  message(FATAL_ERROR "YAKL not found. Aborting.")
endif()