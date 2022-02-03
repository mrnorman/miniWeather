
#pragma once

#include "YAKL.h"
#include <cmath>

using yakl::SArray;
using yakl::c::SimpleBounds;

typedef double real;

inline real operator"" _fp( long double x ) {
  return static_cast<real>(x);
}

constexpr real pi        = 3.14159265358979323846264338327;   //Pi
constexpr real grav      = 9.8;                               //Gravitational acceleration (m / s^2)
constexpr real cp        = 1004.;                             //Specific heat of dry air at constant pressure
constexpr real cv        = 717.;                              //Specific heat of dry air at constant volume
constexpr real rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
constexpr real p0        = 1.e5;                              //Standard pressure at the surface in Pascals
constexpr real C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
constexpr real gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
constexpr real xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
constexpr real zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
constexpr real hv_beta   = 0.05;    //How strong to diffuse the solution: hv_beta \in [0:1]
constexpr real cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
constexpr real max_speed = 450;     //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
constexpr int hs        = 2;        //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
constexpr int sten_size = 4;        //Size of the stencil used for interpolation

//Parameters for indexing and flags
constexpr int NUM_VARS = 4;           //Number of fluid state variables
constexpr int ID_DENS  = 0;           //index for density ("rho")
constexpr int ID_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
constexpr int ID_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
constexpr int ID_RHOT  = 3;           //index for density * potential temperature ("rho * theta")
constexpr int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
constexpr int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
constexpr int DATA_SPEC_COLLISION       = 1;
constexpr int DATA_SPEC_THERMAL         = 2;
constexpr int DATA_SPEC_GRAVITY_WAVES   = 3;
constexpr int DATA_SPEC_DENSITY_CURRENT = 5;
constexpr int DATA_SPEC_INJECTION       = 6;

///////////////////////////////////////////////////////////////////////////////////////
// BEGIN USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////
//The x-direction length is twice as long as the z-direction length
//So, you'll want to have nx_glob be twice as large as nz_glob
int  constexpr nx_glob = _NX;        // Number of total cells in the x-dirction
int  constexpr nz_glob = _NZ;        // Number of total cells in the z-dirction
real constexpr sim_time = _SIM_TIME; // How many seconds to run the simulation
real constexpr output_freq = _OUT_FREQ;  // How frequently to output data to file (in seconds)
int  constexpr data_spec_int = _DATA_SPEC; // How to initialize the data
///////////////////////////////////////////////////////////////////////////////////////
// END USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////
real constexpr dx = xlen / nx_glob;
real constexpr dz = zlen / nz_glob;

using yakl::c::Bounds;
using yakl::c::parallel_for;
using yakl::SArray;

template<class T> inline T min( T val1 , T val2 ) {
  return val1 < val2 ? val1 : val2 ;
}

template<class T> inline T abs( T val ) {
  return val > 0 ? val : -val;
}

#ifdef SIMD_LEN
  int constexpr simd_len = SIMD_LEN;
#else
  int constexpr simd_len = 4;
#endif

using yakl::Pack;

