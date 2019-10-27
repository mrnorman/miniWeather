
#ifndef __CONST_H__
#define __CONST_H__

#include "YAKL.h"
#include <cmath>

#ifdef __NVCC__
  #define _HOSTDEV __host__ __device__
#else
  #define _HOSTDEV 
#endif

#include "SArray.h"
#include "Array.h"

typedef float real;

inline real operator"" _fp( long double x ) {
  return static_cast<real>(x);
}

#if defined(__USE_CUDA__) || defined(__USE_HIP__)
  typedef yakl::Array<real,yakl::memDevice> realArr;
#else
  typedef yakl::Array<real,yakl::memHost> realArr;
#endif
typedef yakl::Array<real,yakl::memHost> realArrHost;

const real pi        = 3.14159265358979323846264338327;   //Pi
const real grav      = 9.8;                               //Gravitational acceleration (m / s^2)
const real cp        = 1004.;                             //Specific heat of dry air at constant pressure
const real cv        = 717.;                              //Specific heat of dry air at constant volume
const real rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const real p0        = 1.e5;                              //Standard pressure at the surface in Pascals
const real C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const real gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const real xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
const real zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
const real hv_beta   = 0.25;     //How strong to diffuse the solution: hv_beta \in [0:1]
const real cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
const real max_speed = 450;        //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
const int sten_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
const int NUM_VARS = 4;           //Number of fluid state variables
const int ID_DENS  = 0;           //index for density ("rho")
const int ID_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
const int ID_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
const int ID_RHOT  = 3;           //index for density * potential temperature ("rho * theta")
const int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
const int DATA_SPEC_COLLISION       = 1;
const int DATA_SPEC_THERMAL         = 2;
const int DATA_SPEC_MOUNTAIN        = 3;
const int DATA_SPEC_TURBULENCE      = 4;
const int DATA_SPEC_DENSITY_CURRENT = 5;
const int DATA_SPEC_INJECTION       = 6;

template<class T> inline T min( T val1 , T val2 ) {
  return val1 < val2 ? val1 : val2 ;
}


#endif

