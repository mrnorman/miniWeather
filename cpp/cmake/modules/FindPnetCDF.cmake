find_library(pnetcdf_lib_found pnetcdf PATHS ${PNETCDF_ROOT}/lib NO_DEFAULT_PATHS)
find_path(pnetcdf_headers_found pnetcdf.h PATHS ${PNETCDF_ROOT}/include NO_DEFAULT_PATHS)

if (pnetcdf_lib_found AND pnetcdf_headers_found)
  add_library(PnetCDF INTERFACE)
  set_target_properties(PnetCDF PROPERTIES
    INTERFACE_LINK_LIBRARIES ${pnetcdf_lib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${pnetcdf_headers_found}
  )
else()
  message(FATAL_ERROR "PnetCDF not found. Aborting.")
endif()