
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
#include "Array.h"

typedef double F;

const F pi        = 3.14159265358979323846264338327;   //Pi
const F grav      = 9.8;                               //Gravitational acceleration (m / s^2)
const F cp        = 1004.;                             //Specific heat of dry air at constant pressure
const F cv        = 717.;                              //Specific heat of dry air at constant volume
const F rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const F p0        = 1.e5;                              //Standard pressure at the surface in Pascals
const F C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const F gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const F xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
const F zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
const F hv_beta   = 0.25;    //How strong to diffuse the solution: hv_beta \in [0:1]
const F cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
const F max_speed = 450;     //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
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

const int nqpoints = 3;
F qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
F qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
F   sim_time;              //total simulation time in seconds
F   output_freq;           //frequency to perform output in seconds
F   dt;                    //Model time step (seconds)
int nx, nz;                //Number of local grid cells in the x- and z- dimensions for this MPI task
F   dx, dz;                //Grid space length in x- and z-dimension (meters)
int nx_glob, nz_glob;      //Number of total grid cells in the x- and z- dimensions
int i_beg, k_beg;          //beginning index in the x- and z-directions for this MPI task
int nranks, myrank;        //Number of MPI ranks and my rank id
int left_rank, right_rank; //MPI Rank IDs that exist to my left and right in the global domain
int masterproc;            //Am I the master process (rank == 0)?
F   data_spec_int;         //Which data initialization to use


Array<F> hy_dens_cell;         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
Array<F> hy_dens_theta_cell;   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
Array<F> hy_dens_int;          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
Array<F> hy_dens_theta_int;    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
Array<F> hy_pressure_int;      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
F etime;                 //Elapsed model time
F output_counter;        //Helps determine when it's time to do output
//Runtime variable arrays
Array<F> state;          //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
Array<F> state_tmp;      //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
Array<F> flux;           //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
Array<F> tend;           //Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
int      num_out = 0;    //The number of outputs performed so far
int      direction_switch = 1;

//How is this not in the standard?!
F dmin( F a , F b ) { if (a<b) {return a;} else {return b;} };


//Declaring the functions defined after "main"
void   init                 ( int *argc , char ***argv );
void   finalize             ( );
void   injection            ( F const x , F const z , F &r , F &u , F &w , F &t , F &hr , F &ht );
void   density_current      ( F const x , F const z , F &r , F &u , F &w , F &t , F &hr , F &ht );
void   turbulence           ( F const x , F const z , F &r , F &u , F &w , F &t , F &hr , F &ht );
void   mountain_waves       ( F const x , F const z , F &r , F &u , F &w , F &t , F &hr , F &ht );
void   thermal              ( F const x , F const z , F &r , F &u , F &w , F &t , F &hr , F &ht );
void   collision            ( F const x , F const z , F &r , F &u , F &w , F &t , F &hr , F &ht );
void   hydro_const_theta    ( F const z , F &r , F &t );
void   hydro_const_bvfreq   ( F const z , F const bv_freq0 , F &r , F &t );
F      sample_ellipse_cosine( F const x , F const z , F const amp , F const x0 , F const z0 , F const xrad , F const zrad );
void   output               ( Array<F> const &state , F const etime );
void   ncwrap               ( int const ierr , int const line );
void   perform_timestep     ( Array<F> &state , Array<F> &state_tmp , Array<F> &flux , Array<F> &tend , F const dt );
void   semi_discrete_step   ( Array<F> const &state_init , Array<F> &state_forcing , Array<F> &state_out , F const dt , int const dir , Array<F> &flux , Array<F> &tend );
void   compute_tendencies_x ( Array<F> const &state , Array<F> &flux , Array<F> &tend );
void   compute_tendencies_z ( Array<F> const &state , Array<F> &flux , Array<F> &tend );
void   set_halo_values_x    ( Array<F> &state );
void   set_halo_values_z    ( Array<F> &state );


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  ///////////////////////////////////////////////////////////////////////////////////////
  // BEGIN USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////
  //The x-direction length is twice as long as the z-direction length
  //So, you'll want to have nx_glob be twice as large as nz_glob
  nx_glob = 200;      //Number of total cells in the x-dirction
  nz_glob = 100;      //Number of total cells in the z-dirction
  sim_time = 1500;     //How many seconds to run the simulation
  output_freq = 10;   //How frequently to output data to file (in seconds)
  //Model setup: DATA_SPEC_THERMAL or DATA_SPEC_COLLISION
  data_spec_int = DATA_SPEC_INJECTION;
  ///////////////////////////////////////////////////////////////////////////////////////
  // END USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////

  init( &argc , &argv );

  //Output the initial state
  output(state,etime);

  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  while (etime < sim_time) {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) { dt = sim_time - etime; }
    //Perform a single time step
    perform_timestep(state,state_tmp,flux,tend,dt);
    //Inform the user
    if (masterproc) { printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); }
    //Update the elapsed time and output counter
    etime = etime + dt;
    output_counter = output_counter + dt;
    //If it's time for output, reset the counter, and do output
    if (output_counter >= output_freq) {
      output_counter = output_counter - output_freq;
      output(state,etime);
    }
  }


  finalize();
}


//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void perform_timestep( Array<F> &state , Array<F> &state_tmp , Array<F> &flux , Array<F> &tend , F const dt ) {
  if (direction_switch) {
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
  } else {
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}


//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step( Array<F> const &state_init , Array<F> &state_forcing , Array<F> &state_out , F const dt , int const dir , Array<F> &flux , Array<F> &tend ) {
  int i, k, ll;
  if        (dir == DIR_X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    set_halo_values_x(state_forcing);
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(state_forcing,flux,tend);
  } else if (dir == DIR_Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    set_halo_values_z(state_forcing);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(state_forcing,flux,tend);
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        state_out(i+hs,k+hs,ll) = state_init(i+hs,k+hs,ll) + dt * tend(i,k,ll);
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x( Array<F> const &state , Array<F> &flux , Array<F> &tend ) {
  int i,k,ll,s;
  F   r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  F   *state_d, *flux_d;
  int state_n, flux_n;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dx / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  state_d = state.get_data();
  state_n = state.get_totElems();
  flux_d = state.get_data();
  flux_n = state.get_totElems();
  #pragma data copyin(state_d[0:state_n]) copyout(flux_d[0:flux_n])
  {
  #pragma acc parallel loop gang vector collapse(2) private(vals,d3_vals,stencil) default(present)
  for (k=0; k<nz; k++) {
    for (i=0; i<nx+1; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s < sten_size; s++) {
          stencil[s] = state(i+s,k+hs,ll);
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_cell(k+hs);
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_cell(k+hs) ) / r;
      p = C0*pow((r*t),gamm);

      //Compute the flux vector
      flux(i,k,ID_DENS) = r*u     - hv_coef*d3_vals[ID_DENS];
      flux(i,k,ID_UMOM) = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
      flux(i,k,ID_WMOM) = r*u*w   - hv_coef*d3_vals[ID_WMOM];
      flux(i,k,ID_RHOT) = r*u*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        tend(i,k,ll) = -( flux(i+1,k,ll) - flux(i,k,ll) ) / dx;
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z( Array<F> const &state , Array<F> &flux , Array<F> &tend ) {
  int i,k,ll,s;
  F   r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dx / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nz+1; k++) {
    for (i=0; i<nx; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s<sten_size; s++) {
          stencil[s] = state(i+hs,k+s,ll);
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_int(k);
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_int(k) ) / r;
      p = C0*pow((r*t),gamm) - hy_pressure_int(k);

      //Compute the flux vector with hyperviscosity
      flux(i,k,ID_DENS) = r*w     - hv_coef*d3_vals[ID_DENS];
      flux(i,k,ID_UMOM) = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      flux(i,k,ID_WMOM) = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      flux(i,k,ID_RHOT) = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        tend(i,k,ll) = -( flux(i,k+1,ll) - flux(i,k,ll) ) / dz;
        if (ll == ID_WMOM) {
          tend(i,k,ll) = tend(i,k,ll) - state(i+hs,k+hs,ID_DENS)*grav;
        }
      }
    }
  }
}



//Set this MPI task's halo values in the x-direction. This routine will require MPI
void set_halo_values_x( Array<F> &state ) {
  int k, ll, i;
  F   z;
  ////////////////////////////////////////////////////////////////////////
  // TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  // (1) give    state(1:hs,1:nz,1:NUM_VARS)       to   my left  neighbor
  // (2) receive state(1-hs:0,1:nz,1:NUM_VARS)     from my left  neighbor
  // (3) give    state(nx-hs+1:nx,1:nz,1:NUM_VARS) to   my right neighbor
  // (4) receive state(nx+1:nx+hs,1:nz,1:NUM_VARS) from my right neighbor
  ////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  //////////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      state(0      ,k+hs,ll) = state(nx+hs-2,k+hs,ll);
      state(1      ,k+hs,ll) = state(nx+hs-1,k+hs,ll);
      state(nx+hs  ,k+hs,ll) = state(hs     ,k+hs,ll);
      state(nx+hs+1,k+hs,ll) = state(hs+1   ,k+hs,ll);
    }
  }
  ////////////////////////////////////////////////////

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      for (k=0; k<nz; k++) {
        for (i=0; i<hs; i++) {
          z = (k_beg + k+0.5)*dz;
          if (abs(z-3*zlen/4) <= zlen/16) {
            state(i,k+hs,ID_UMOM) = (state(i,k+hs,ID_DENS)+hy_dens_cell(k+hs)) * 50.;
            state(i,k+hs,ID_RHOT) = (state(i,k+hs,ID_DENS)+hy_dens_cell(k+hs)) * 298. - hy_dens_theta_cell(k+hs);
          }
        }
      }
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void set_halo_values_z( Array<F> &state ) {
  int     i, ll;
  const F mnt_width = xlen/8;
  F       x, xloc, mnt_deriv;
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (i=0; i<nx+2*hs; i++) {
      if (ll == ID_WMOM) {
        state(i,0      ,ll) = 0.;
        state(i,1      ,ll) = 0.;
        state(i,nz+hs  ,ll) = 0.;
        state(i,nz+hs+1,ll) = 0.;
        //Impose the vertical momentum effects of an artificial cos^2 mountain at the lower boundary
        if (data_spec_int == DATA_SPEC_MOUNTAIN) {
          x = (i_beg+i-hs+0.5)*dx;
          if ( fabs(x-xlen/4) < mnt_width ) {
            xloc = (x-(xlen/4)) / mnt_width;
            //Compute the derivative of the fake mountain
            mnt_deriv = -pi*cos(pi*xloc/2)*sin(pi*xloc/2)*10/dx;
            //w = (dz/dx)*u
            state(i,0,ID_WMOM) = mnt_deriv*state(i,hs,ID_UMOM);
            state(i,1,ID_WMOM) = mnt_deriv*state(i,hs,ID_UMOM);
          }
        }
      } else {
        state(i,0      ,ll) = state(i,hs     ,ll);
        state(i,1      ,ll) = state(i,hs     ,ll);
        state(i,nz+hs  ,ll) = state(i,nz+hs-1,ll);
        state(i,nz+hs+1,ll) = state(i,nz+hs-1,ll);
      }
    }
  }
}


void init( int *argc , char ***argv ) {
  int    i, k, ii, kk, ll, ierr;
  F x, z, r, u, w, t, hr, ht;

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
  state             .setup(nx+2*hs,nz+2*hs,NUM_VARS);
  state_tmp         .setup(nx+2*hs,nz+2*hs,NUM_VARS);
  flux              .setup(nx+1,nz+1,NUM_VARS);
  tend              .setup(nx,nz,NUM_VARS);
  hy_dens_cell      .setup(nz+2*hs);
  hy_dens_theta_cell.setup(nz+2*hs);
  hy_dens_int       .setup(nz+1);
  hy_dens_theta_int .setup(nz+1);
  hy_pressure_int   .setup(nz+1);

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
        state(i,k,ll) = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          //Store into the fluid state array
          state(i,k,ID_DENS) = state(i,k,ID_DENS) + r                         * qweights[ii]*qweights[kk];
          state(i,k,ID_UMOM) = state(i,k,ID_UMOM) + (r+hr)*u                  * qweights[ii]*qweights[kk];
          state(i,k,ID_WMOM) = state(i,k,ID_WMOM) + (r+hr)*w                  * qweights[ii]*qweights[kk];
          state(i,k,ID_RHOT) = state(i,k,ID_RHOT) + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        state_tmp(i,k,ll) = state(i,k,ll);
      }
    }
  }
  //Compute the hydrostatic background state over vertical cell averages
  for (k=0; k<nz+2*hs; k++) {
    hy_dens_cell      (k) = 0.;
    hy_dens_theta_cell(k) = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      (k) = hy_dens_cell      (k) + hr    * qweights[kk];
      hy_dens_theta_cell(k) = hy_dens_theta_cell(k) + hr*ht * qweights[kk];
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
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      (k) = hr;
    hy_dens_theta_int(k) = hr*ht;
    hy_pressure_int  (k) = C0*pow((hr*ht),gamm);
  }
}


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void injection( F x , F z , F &r , F &u , F &w , F &t , F &hr , F &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void density_current( F x , F z , F &r , F &u , F &w , F &t , F &hr , F &ht ) {
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
void turbulence( F x , F z , F &r , F &u , F &w , F &t , F &hr , F &ht ) {
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
void mountain_waves( F x , F z , F &r , F &u , F &w , F &t , F &hr , F &ht ) {
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
void thermal( F x , F z , F &r , F &u , F &w , F &t , F &hr , F &ht ) {
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
void collision( F x , F z , F &r , F &u , F &w , F &t , F &hr , F &ht ) {
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
void hydro_const_theta( F z , F &r , F &t ) {
  const F theta0 = 300.;  //Background potential temperature
  const F exner0 = 1.;    //Surface-level Exner pressure
  F       p,exner,rt;
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
void hydro_const_bvfreq( F z , F bv_freq0 , F &r , F &t ) {
  const F theta0 = 300.;  //Background potential temperature
  const F exner0 = 1.;    //Surface-level Exner pressure
  F       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
F sample_ellipse_cosine( F x , F z , F amp , F x0 , F z0 , F xrad , F zrad ) {
  F dist;
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
void output( Array<F> const &state , F etime ) {
  int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid, dimids[3];
  int i, k;
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];
  //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
  F *dens, *uwnd, *wwnd, *theta;
  F *etimearr;
  //Inform the user
  if (masterproc) { printf("*** OUTPUT ***\n"); }
  //Allocate some (big) temp arrays
  dens     = (F *) malloc(nx*nz*sizeof(F));
  uwnd     = (F *) malloc(nx*nz*sizeof(F));
  wwnd     = (F *) malloc(nx*nz*sizeof(F));
  theta    = (F *) malloc(nx*nz*sizeof(F));
  etimearr = (F *) malloc(1    *sizeof(F));

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
      dens [k*nx+i] = state(i+hs,k+hs,ID_DENS);
      uwnd [k*nx+i] = state(i+hs,k+hs,ID_UMOM) / ( hy_dens_cell(k+hs) + state(i+hs,k+hs,ID_DENS) );
      wwnd [k*nx+i] = state(i+hs,k+hs,ID_WMOM) / ( hy_dens_cell(k+hs) + state(i+hs,k+hs,ID_DENS) );
      theta[k*nx+i] = ( state(i+hs,k+hs,ID_RHOT) + hy_dens_theta_cell(k+hs) ) / ( hy_dens_cell(k+hs) + state(i+hs,k+hs,ID_DENS) ) - hy_dens_theta_cell(k+hs) / hy_dens_cell(k+hs);
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
  state.finalize();
  state_tmp.finalize();
  flux.finalize();
  tend.finalize();
  hy_dens_cell.finalize();
  hy_dens_theta_cell.finalize();
  hy_dens_int.finalize();
  hy_dens_theta_int.finalize();
  hy_pressure_int.finalize();
  ierr = MPI_Finalize();
}