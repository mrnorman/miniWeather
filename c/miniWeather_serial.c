
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "pnetcdf.h"

const double pi        = 3.14159265358979323846264338327;   //Pi
const double grav      = 9.8;                               //Gravitational acceleration (m / s^2)
const double cp        = 1004.;                             //Specific heat of dry air at constant pressure
const double cv        = 717.;                              //Specific heat of dry air at constant volume
const double rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const double p0        = 1.e5;                              //Standard pressure at the surface in Pascals
const double C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const double gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const double xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
const double zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
const double hv_beta   = 0.25;     //How strong to diffuse the solution: hv_beta \in [0:1]
const double cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
const double max_speed = 450;        //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
const int sten_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
const int NUM_VARS = 4;           //Number of fluid state variables
const int ID_DENS  = 1;           //index for density ("rho")
const int ID_UMOM  = 2;           //index for momentum in the x-direction ("rho * u")
const int ID_WMOM  = 3;           //index for momentum in the z-direction ("rho * w")
const int ID_RHOT  = 4;           //index for density * potential temperature ("rho * theta")
const int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
const int DATA_SPEC_COLLISION       = 1;
const int DATA_SPEC_THERMAL         = 2;
const int DATA_SPEC_MOUNTAIN        = 3;
const int DATA_SPEC_TURBULENCE      = 4;
const int DATA_SPEC_DENSITY_CURRENT = 5;

const int nqpoints = 3;
double qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
double qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double sim_time;              //total simulation time in seconds
double output_freq;           //frequency to perform output in seconds
double dt;                    //Model time step (seconds)
int    nx, nz;                //Number of local grid cells in the x- and z- dimensions for this MPI task
double dx, dz;                //Grid space length in x- and z-dimension (meters)
int    nx_glob, nz_glob;      //Number of total grid cells in the x- and z- dimensions
int    i_beg, k_beg;          //beginning index in the x- and z-directions for this MPI task
int    nranks, myrank;        //Number of MPI ranks and my rank id
int    left_rank, right_rank; //MPI Rank IDs that exist to my left and right in the global domain
int    masterproc;            //Am I the master process (rank == 0)?
double data_spec_int;         //Which data initialization to use
double *hy_dens_cell;         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
double *hy_dens_theta_cell;   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
double *hy_dens_int;          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
double *hy_dens_theta_int;    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
double *hy_pressure_int;      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;                 //Elapsed model time
double output_counter;        //Helps determine when it's time to do output
//Runtime variable arrays
double *state;                //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
double *state_tmp;            //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
double *flux;                 //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
double *tend;                 //Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
int    num_out = 0;           //The number of outputs performed so far

double dmin( double a , double b ) { if (a<b) {return a;} else {return b;} };

void   init( int *argc , char ***argv );
void   finalize( );
void   density_current( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   turbulence     ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   mountain_waves ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   thermal        ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   collision      ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   hydro_const_theta ( double z                   , double &r , double &t );
void   hydro_const_bvfreq( double z , double bv_freq0 , double &r , double &t );
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad );
void   output( double *state , double etime );
void   ncwrap( int ierr , int line );

int main(int argc, char **argv) {
  int ierr;
  ///////////////////////////////////////////////////////////////////////////////////////
  // BEGIN USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////
  //The x-direction length is twice as long as the z-direction length
  //So, you'll want to have nx_glob be twice as large as nz_glob
  nx_glob = 200;      //Number of total cells in the x-dirction
  nz_glob = 100;      //Number of total cells in the z-dirction
  sim_time = 2000;    //How many seconds to run the simulation
  output_freq = 10;   //How frequently to output data to file (in seconds)
  //Model setup: DATA_SPEC_THERMAL or DATA_SPEC_COLLISION
  data_spec_int = DATA_SPEC_DENSITY_CURRENT;
  ///////////////////////////////////////////////////////////////////////////////////////
  // END USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////

  init( &argc , &argv );

  //Output the initial state
  output(state,etime);

  finalize();
}


void init( int *argc , char ***argv ) {
  int    i, k, ii, kk, ll, ierr, inds;
  double x, z, r, u, w, t, hr, ht;

  ierr = MPI_Init(argc,argv);

  //Set the cell grid size
  dx = xlen / nx_glob;
  dz = zlen / nz_glob;

  /////////////////////////////////////////////////////////////
  // BEGIN MPI DUMMY SECTION
  // TODO: (1) GET NUMBER OF MPI RANKS
  //       (2) GET MY MPI RANK ID (RANKS ARE ZERO-BASED INDEX)
  //       (3) COMPUTE MY BEGINNING "I" INDEX (1-based index)
  //       (4) COMPUTE HOW MANY X-DIRECTION CELLS MY RANK HAS
  //       (5) FIND MY LEFT AND RIGHT NEIGHBORING RANK IDs
  /////////////////////////////////////////////////////////////
  nranks = 1;
  myrank = 0;
  i_beg = 0;
  nx = nx_glob;
  left_rank = 0;
  right_rank = 0;
  //////////////////////////////////////////////
  // END MPI DUMMY SECTION
  //////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  k_beg = 0;
  nz = nz_glob;
  masterproc = (myrank == 0);

  //Allocate the model data
  state              = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  state_tmp          = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  flux               = (double *) malloc( (nx+1)*(nz+1)*NUM_VARS*sizeof(double) );
  tend               = (double *) malloc( nx*nz*NUM_VARS*sizeof(double) );
  hy_dens_cell       = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_theta_cell = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_int        = (double *) malloc( (nz+1)*sizeof(double) );
  hy_dens_theta_int  = (double *) malloc( (nz+1)*sizeof(double) );
  hy_pressure_int    = (double *) malloc( (nz+1)*sizeof(double) );

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = dmin(dx,dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  //If I'm the master process in MPI, display some grid information
  if (masterproc) {
    printf( "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }
  //Want to make sure this info is displayed before further output
  ierr = MPI_Barrier(MPI_COMM_WORLD);

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k=0; k<nz+2*hs; k++) {
    for (i=0; i<nx+2*hs; i++) {
      //Initialize the state to zero
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state[inds] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }

          //Store into the fluid state array
          inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + r                         * qweights[ii]*qweights[kk];
          inds = ID_UMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*u                  * qweights[ii]*qweights[kk];
          inds = ID_WMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*w                  * qweights[ii]*qweights[kk];
          inds = ID_RHOT*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state_tmp[inds] = state[inds];
      }
    }
  }
  //Compute the hydrostatic background state over vertical cell averages
  for (k=0; k<nz+2*hs; k++) {
    hy_dens_cell      [k] = 0.;
    hy_dens_theta_cell[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      [k] = hy_dens_cell      [k] + hr    * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  for (k=0; k<nz+1; k++) {
    z = (k_beg + k)*dz;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      [k] = hr;
    hy_dens_theta_int[k] = hr*ht;
    hy_pressure_int  [k] = C0*pow((hr*ht),gamm);
  }
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void density_current( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void turbulence( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  // call random_number(u);
  // call random_number(w);
  // u = (u-0.5)*20;
  // w = (w-0.5)*20;
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void mountain_waves( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_bvfreq(z,0.02,hr,ht);
  r = 0.;
  t = 0.;
  u = 15.;
  w = 0.;
}


//Rising thermal
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void thermal( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);
}


//Colliding thermals
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void collision( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}


//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_theta( double z , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrstatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_bvfreq( double z , double bv_freq0 , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}


  //Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
  //The file I/O uses parallel-netcdf, the only external library required for this mini-app.
  //If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
  void output( double *state , double etime ) {
    int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid, dimids[3];
    int i, k, ind_r, ind_u, ind_w, ind_t;
    MPI_Offset len, st1[1], ct1[1], st3[3], ct3[3];
    //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
    double *dens, *uwnd, *wwnd, *theta;
    double *etimearr;
    //Inform the user
    if (masterproc) { printf("*** OUTPUT ***\n"); }
    //Allocate some (big) temp arrays
    dens     = (double *) malloc(nx*nz*sizeof(double));
    uwnd     = (double *) malloc(nx*nz*sizeof(double));
    wwnd     = (double *) malloc(nx*nz*sizeof(double));
    theta    = (double *) malloc(nx*nz*sizeof(double));
    etimearr = (double *) malloc(1    *sizeof(double));

    //If the elapsed time is zero, create the file. Otherwise, open the file
    if (etime == 0) {
      //Create the file
      ncwrap( ncmpi_create( MPI_COMM_WORLD , "output.nc" , NC_CLOBBER , MPI_INFO_NULL , &ncid ) , __LINE__ );
      //Create the dimensions
      ncwrap( ncmpi_def_dim( ncid , "t" , (MPI_Offset) NC_UNLIMITED , &t_dimid ) , __LINE__ );
      ncwrap( ncmpi_def_dim( ncid , "x" , (MPI_Offset) nx_glob      , &x_dimid ) , __LINE__ );
      ncwrap( ncmpi_def_dim( ncid , "z" , (MPI_Offset) nz_glob      , &z_dimid ) , __LINE__ );
      //Create the variables
      dimids[0] = t_dimid;
      ncwrap( ncmpi_def_var( ncid , "t"     , NC_DOUBLE , 1 , dimids ,     &t_varid ) , __LINE__ );
      dimids[0] = t_dimid; dimids[1] = z_dimid; dimids[2] = x_dimid;
      ncwrap( ncmpi_def_var( ncid , "dens"  , NC_DOUBLE , 3 , dimids ,  &dens_varid ) , __LINE__ );
      ncwrap( ncmpi_def_var( ncid , "uwnd"  , NC_DOUBLE , 3 , dimids ,  &uwnd_varid ) , __LINE__ );
      ncwrap( ncmpi_def_var( ncid , "wwnd"  , NC_DOUBLE , 3 , dimids ,  &wwnd_varid ) , __LINE__ );
      ncwrap( ncmpi_def_var( ncid , "theta" , NC_DOUBLE , 3 , dimids , &theta_varid ) , __LINE__ );
      //End "define" mode
      ncwrap( ncmpi_enddef( ncid ) , __LINE__ );
    } else {
      //Open the file
      ncwrap( ncmpi_open( MPI_COMM_WORLD , "output.nc" , NC_WRITE , MPI_INFO_NULL , &ncid ) , __LINE__ );
      //Get the variable IDs
      ncwrap( ncmpi_inq_varid( ncid , "dens"  ,  &dens_varid ) , __LINE__ );
      ncwrap( ncmpi_inq_varid( ncid , "uwnd"  ,  &uwnd_varid ) , __LINE__ );
      ncwrap( ncmpi_inq_varid( ncid , "wwnd"  ,  &wwnd_varid ) , __LINE__ );
      ncwrap( ncmpi_inq_varid( ncid , "theta" , &theta_varid ) , __LINE__ );
      ncwrap( ncmpi_inq_varid( ncid , "t"     ,     &t_varid ) , __LINE__ );
    }

    //Store perturbed values in the temp arrays for output
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        dens [k*nx+i] = state[ind_r];
        uwnd [k*nx+i] = state[ind_u] / ( hy_dens_cell[k+hs] + state[ind_r] );
        wwnd [k*nx+i] = state[ind_w] / ( hy_dens_cell[k+hs] + state[ind_r] );
        theta[k*nx+i] = ( state[ind_t] + hy_dens_theta_cell[k+hs] ) / ( hy_dens_cell[k+hs] + state[ind_r] ) - hy_dens_theta_cell[k+hs] / hy_dens_cell[k+hs];
      }
    }

    //Write the grid data to file with all the processes writing collectively
    st3[0] = num_out; st3[1] = k_beg; st3[2] = i_beg;
    ct3[0] = 1      ; ct3[1] = nz   ; ct3[2] = nx   ;
    ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens  ) , __LINE__ );
    ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd  ) , __LINE__ );
    ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd  ) , __LINE__ );
    ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta ) , __LINE__ );

    //Only the master process needs to write the elapsed time
    //Begin "independent" write mode
    ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
    //write elapsed time to file
    if (masterproc) {
      st1[0] = num_out;
      ct1[0] = 1;
      etimearr[0] = etime; ncwrap( ncmpi_put_vara_double( ncid , t_varid , st1 , ct1 , etimearr ) , __LINE__ );
    }
    //End "independent" write mode
    ncwrap( ncmpi_end_indep_data(ncid) , __LINE__ );

    //Close the file
    ncwrap( ncmpi_close(ncid) , __LINE__ );

    //Increment the number of outputs
    num_out = num_out + 1;

    //Deallocate the temp arrays
    free( dens     );
    free( uwnd     );
    free( wwnd     );
    free( theta    );
    free( etimearr );
  }


  //Error reporting routine for the PNetCDF I/O
  void ncwrap( int ierr , int line ) {
    if (ierr != NC_NOERR) {
      printf("NetCDF Error at line: %d\n", line);
      printf("%s\n",ncmpi_strerror(ierr));
      exit(-1);
    }
  }


  void finalize() {
    int ierr;
    free( state );
    free( state_tmp );
    free( flux );
    free( tend );
    free( hy_dens_cell );
    free( hy_dens_theta_cell );
    free( hy_dens_int );
    free( hy_dens_theta_int );
    free( hy_pressure_int );
    ierr = MPI_Finalize();
  }
