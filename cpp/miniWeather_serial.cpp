
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <ctime>
#include "const.h"
#include "pnetcdf.h"

// We're going to define all arrays on the host because this doesn't use parallel_for
typedef yakl::Array<real  ,1,yakl::memHost> real1d;
typedef yakl::Array<real  ,2,yakl::memHost> real2d;
typedef yakl::Array<real  ,3,yakl::memHost> real3d;
typedef yakl::Array<double,1,yakl::memHost> doub1d;
typedef yakl::Array<double,2,yakl::memHost> doub2d;
typedef yakl::Array<double,3,yakl::memHost> doub3d;

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
real sim_time;              //total simulation time in seconds
real output_freq;           //frequency to perform output in seconds
real dt;                    //Model time step (seconds)
int nx, nz;                 //Number of local grid cells in the x- and z- dimensions for this MPI task
real dx, dz;                //Grid space length in x- and z-dimension (meters)
int nx_glob, nz_glob;       //Number of total grid cells in the x- and z- dimensions
int i_beg, k_beg;           //beginning index in the x- and z-directions for this MPI task
int nranks, myrank;         //Number of MPI ranks and my rank id
int left_rank, right_rank;  //MPI Rank IDs that exist to my left and right in the global domain
int masterproc;             //Am I the master process (rank == 0)?
real data_spec_int;         //Which data initialization to use
real1d hy_dens_cell;        //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
real1d hy_dens_theta_cell;  //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
real1d hy_dens_int;         //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
real1d hy_dens_theta_int;   //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
real1d hy_pressure_int;     //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
real etime;                 //Elapsed model time
real output_counter;        //Helps determine when it's time to do output
//Runtime variable arrays
real3d state;               //Fluid state.             Dimensions: (NUM_VARS,1-hs:nz+hs,1-hs:nx+hs)
real3d state_tmp;           //Fluid state.             Dimensions: (NUM_VARS,1-hs:nz+hs,1-hs:nx+hs)
real3d flux;                //Cell interface fluxes.   Dimensions: (NUM_VARS,nz+1,nx+1)
real3d tend;                //Fluid state tendencies.  Dimensions: (NUM_VARS,nz,nx)
int  num_out = 0;           //The number of outputs performed so far
int  direction_switch = 1;
double mass0, te0;            //Initial domain totals for mass and total energy  
double mass , te ;            //Domain totals for mass and total energy  


//Declaring the functions defined after "main"
void init                 ( int *argc , char ***argv );
void finalize             ( );
void injection            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void density_current      ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void turbulence           ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void mountain_waves       ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void thermal              ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void collision            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void hydro_const_theta    ( real z                    , real &r , real &t );
void hydro_const_bvfreq   ( real z , real bv_freq0    , real &r , real &t );
real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad );
void output               ( real3d &state , real etime );
void ncwrap               ( int ierr , int line );
void perform_timestep     ( real3d &state , real3d &state_tmp , real3d &flux , real3d &tend , real dt );
void semi_discrete_step   ( real3d &state_init , real3d &state_forcing , real3d &state_out , real dt , int dir , real3d &flux , real3d &tend );
void compute_tendencies_x ( real3d &state , real3d &flux , real3d &tend );
void compute_tendencies_z ( real3d &state , real3d &flux , real3d &tend );
void set_halo_values_x    ( real3d &state );
void set_halo_values_z    ( real3d &state );
void reductions           ( double &mass , double &te );


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  yakl::init();
  {
    ///////////////////////////////////////////////////////////////////////////////////////
    // BEGIN USER-CONFIGURABLE PARAMETERS
    ///////////////////////////////////////////////////////////////////////////////////////
    //The x-direction length is twice as long as the z-direction length
    //So, you'll want to have nx_glob be twice as large as nz_glob
    nx_glob = _NX;        // Number of total cells in the x-dirction
    nz_glob = _NZ;        // Number of total cells in the z-dirction
    sim_time = _SIM_TIME; // How many seconds to run the simulation
    output_freq = _OUT_FREQ;  // How frequently to output data to file (in seconds)
    data_spec_int = _DATA_SPEC; // How to initialize the data
    ///////////////////////////////////////////////////////////////////////////////////////
    // END USER-CONFIGURABLE PARAMETERS
    ///////////////////////////////////////////////////////////////////////////////////////

    init( &argc , &argv );

    //Initial reductions for mass, kinetic energy, and total energy
    reductions(mass0,te0);

    //Output the initial state
    output(state,etime);

    ////////////////////////////////////////////////////
    // MAIN TIME STEP LOOP
    ////////////////////////////////////////////////////
    auto c_start = std::clock();
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
    auto c_end = std::clock();
    if (masterproc) {
      std::cout << "CPU Time: " << ( (double) (c_end-c_start) ) / CLOCKS_PER_SEC << " sec\n";
    }

    //Final reductions for mass, kinetic energy, and total energy
    reductions(mass,te);

    if (masterproc) {
      printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
      printf( "d_te:   %le\n" , (te   - te0  )/te0   );
    }

    finalize();
  }
  yakl::finalize();
}


//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q_n + dt/3 * rhs(q_n)
// q**    = q_n + dt/2 * rhs(q* )
// q_n+1  = q_n + dt/1 * rhs(q**)
void perform_timestep( real3d &state , real3d &state_tmp , real3d &flux , real3d &tend , real dt ) {
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
void semi_discrete_step( real3d &state_init , real3d &state_forcing , real3d &state_out , real dt , int dir , real3d &flux , real3d &tend ) {
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
  // TODO: MAKE THESE 3 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        state_out(ll,hs+k,hs+i) = state_init(ll,hs+k,hs+i) + dt * tend(ll,k,i);
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x( real3d &state , real3d &flux , real3d &tend ) {
  int  i,k,ll,s;
  real r,u,w,t,p, hv_coef;
  SArray<real,1,4> stencil;
  SArray<real,1,NUM_VARS> d3_vals;
  SArray<real,1,NUM_VARS> vals;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dx / (16*dt);
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nz; k++) {
    for (i=0; i<nx+1; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s < sten_size; s++) {
          stencil(s) = state(ll,hs+k,i+s);
        }
        //Fourth-order-accurate interpolation of the state
        vals(ll) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals(ID_DENS) + hy_dens_cell(hs+k);
      u = vals(ID_UMOM) / r;
      w = vals(ID_WMOM) / r;
      t = ( vals(ID_RHOT) + hy_dens_theta_cell(hs+k) ) / r;
      p = C0*pow((r*t),gamm);

      //Compute the flux vector
      flux(ID_DENS,k,i) = r*u     - hv_coef*d3_vals(ID_DENS);
      flux(ID_UMOM,k,i) = r*u*u+p - hv_coef*d3_vals(ID_UMOM);
      flux(ID_WMOM,k,i) = r*u*w   - hv_coef*d3_vals(ID_WMOM);
      flux(ID_RHOT,k,i) = r*u*t   - hv_coef*d3_vals(ID_RHOT);
    }
  }

  /////////////////////////////////////////////////
  // TODO: MAKE THESE 3 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        tend(ll,k,i) = -( flux(ll,k,i+1) - flux(ll,k,i) ) / dx;
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z( real3d &state , real3d &flux , real3d &tend ) {
  int  i,k,ll,s;
  real r,u,w,t,p, hv_coef;
  SArray<real,1,4> stencil;
  SArray<real,1,NUM_VARS> d3_vals;
  SArray<real,1,NUM_VARS> vals;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dz / (16*dt);
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nz+1; k++) {
    for (i=0; i<nx; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s<sten_size; s++) {
          stencil(s) = state(ll,k+s,hs+i);
        }
        //Fourth-order-accurate interpolation of the state
        vals(ll) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals(ID_DENS) + hy_dens_int(k);
      u = vals(ID_UMOM) / r;
      w = vals(ID_WMOM) / r;
      t = ( vals(ID_RHOT) + hy_dens_theta_int(k) ) / r;
      p = C0*pow((r*t),gamm) - hy_pressure_int(k);
      if (k == 0 || k == nz) {
        w                = 0;
        d3_vals(ID_DENS) = 0;
      }

      //Compute the flux vector with hyperviscosity
      flux(ID_DENS,k,i) = r*w     - hv_coef*d3_vals(ID_DENS);
      flux(ID_UMOM,k,i) = r*w*u   - hv_coef*d3_vals(ID_UMOM);
      flux(ID_WMOM,k,i) = r*w*w+p - hv_coef*d3_vals(ID_WMOM);
      flux(ID_RHOT,k,i) = r*w*t   - hv_coef*d3_vals(ID_RHOT);
    }
  }

  //Use the fluxes to compute tendencies for each cell
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 3 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        tend(ll,k,i) = -( flux(ll,k+1,i) - flux(ll,k,i) ) / dz;
        if (ll == ID_WMOM) {
          tend(ll,k,i) -= state(ID_DENS,hs+k,hs+i)*grav;
        }
      }
    }
  }
}



//Set this MPI task's halo values in the x-direction. This routine will require MPI
void set_halo_values_x( real3d &state ) {
  int k, ll, i;
  real z;
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
      state(ll,hs+k,0      ) = state(ll,hs+k,nx+hs-2);
      state(ll,hs+k,1      ) = state(ll,hs+k,nx+hs-1);
      state(ll,hs+k,nx+hs  ) = state(ll,hs+k,hs     );
      state(ll,hs+k,nx+hs+1) = state(ll,hs+k,hs+1   );
    }
  }
  ////////////////////////////////////////////////////

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      /////////////////////////////////////////////////
      // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
      /////////////////////////////////////////////////
      for (k=0; k<nz; k++) {
        for (i=0; i<hs; i++) {
          z = (k_beg + k+0.5)*dz;
          if (abs(z-3*zlen/4) <= zlen/16) {
            state(ID_UMOM,hs+k,i) = (state(ID_DENS,hs+k,i)+hy_dens_cell(hs+k)) * 50.;
            state(ID_RHOT,hs+k,i) = (state(ID_DENS,hs+k,i)+hy_dens_cell(hs+k)) * 298. - hy_dens_theta_cell(hs+k);
          }
        }
      }
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void set_halo_values_z( real3d &state ) {
  int    i, ll;
  const real mnt_width = xlen/8;
  real       x, xloc, mnt_deriv;
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (i=0; i<nx+2*hs; i++) {
      if (ll == ID_WMOM) {
        state(ll,0      ,i) = 0.;
        state(ll,1      ,i) = 0.;
        state(ll,nz+hs  ,i) = 0.;
        state(ll,nz+hs+1,i) = 0.;
        //Impose the vertical momentum effects of an artificial cos^2 mountain at the lower boundary
        if (data_spec_int == DATA_SPEC_MOUNTAIN) {
          x = (i_beg+i-hs+0.5)*dx;
          if ( fabs(x-xlen/4) < mnt_width ) {
            xloc = (x-(xlen/4)) / mnt_width;
            //Compute the derivative of the fake mountain
            mnt_deriv = -pi*cos(pi*xloc/2)*sin(pi*xloc/2)*10/dx;
            //w = (dz/dx)*u
            state(ID_WMOM,0,i) = mnt_deriv*state(ID_UMOM,hs,i);
            state(ID_WMOM,1,i) = mnt_deriv*state(ID_UMOM,hs,i);
          }
        }
      } else {
        state(ll,0      ,i) = state(ll,hs     ,i);
        state(ll,1      ,i) = state(ll,hs     ,i);
        state(ll,nz+hs  ,i) = state(ll,nz+hs-1,i);
        state(ll,nz+hs+1,i) = state(ll,nz+hs-1,i);
      }
    }
  }
}


void init( int *argc , char ***argv ) {
  int  i, k, ii, kk, ll, ierr;
  real x, z, r, u, w, t, hr, ht;

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

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  k_beg = 0;
  nz = nz_glob;
  masterproc = (myrank == 0);

  //Allocate the model data
  state              = real3d( "state"     , NUM_VARS,nz+2*hs,nx+2*hs);
  state_tmp          = real3d( "state_tmp" , NUM_VARS,nz+2*hs,nx+2*hs);
  flux               = real3d( "flux"      , NUM_VARS,nz+1   ,nx+1   );
  tend               = real3d( "tend"      , NUM_VARS,nz     ,nx     );
  hy_dens_cell       = real1d( "hy_dens_cell"       ,  nz+2*hs );
  hy_dens_theta_cell = real1d( "hy_dens_theta_cell" ,  nz+2*hs );
  hy_dens_int        = real1d( "hy_dens_int"        ,  nz+1    );
  hy_dens_theta_int  = real1d( "hy_dens_theta_int"  ,  nz+1    );
  hy_pressure_int    = real1d( "hy_pressure_int"    ,  nz+1    );

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = min(dx,dz) / max_speed * cfl;
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

  // Define quadrature weights and points
  const int nqpoints = 3;
  SArray<real,1,nqpoints> qpoints;
  SArray<real,1,nqpoints> qweights;

  qpoints(0) = 0.112701665379258311482073460022;
  qpoints(1) = 0.500000000000000000000000000000;
  qpoints(2) = 0.887298334620741688517926539980;

  qweights(0) = 0.277777777777777777777777777779;
  qweights(1) = 0.444444444444444444444444444444;
  qweights(2) = 0.277777777777777777777777777779;

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (k=0; k<nz+2*hs; k++) {
    for (i=0; i<nx+2*hs; i++) {
      //Initialize the state to zero
      for (ll=0; ll<NUM_VARS; ll++) {
        state(ll,k,i) = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i-hs+0.5)*dx + (qpoints(ii)-0.5)*dx;
          z = (k_beg + k-hs+0.5)*dz + (qpoints(kk)-0.5)*dz;

          //Set the fluid state based on the user's specification
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          //Store into the fluid state array
          state(ID_DENS,k,i) += r                         * qweights(ii)*qweights(kk);
          state(ID_UMOM,k,i) += (r+hr)*u                  * qweights(ii)*qweights(kk);
          state(ID_WMOM,k,i) += (r+hr)*w                  * qweights(ii)*qweights(kk);
          state(ID_RHOT,k,i) += ( (r+hr)*(t+ht) - hr*ht ) * qweights(ii)*qweights(kk);
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        state_tmp(ll,k,i) = state(ll,k,i);
      }
    }
  }
  //Compute the hydrostatic background state over vertical cell averages
  /////////////////////////////////////////////////
  // TODO: MAKE THIS LOOP A PARALLEL_FOR
  /////////////////////////////////////////////////
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
      hy_dens_cell      (k) = hy_dens_cell      (k) + hr    * qweights(kk);
      hy_dens_theta_cell(k) = hy_dens_theta_cell(k) + hr*ht * qweights(kk);
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  /////////////////////////////////////////////////
  // TODO: MAKE THIS LOOP A PARALLEL_FOR
  /////////////////////////////////////////////////
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
void injection( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
void density_current( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
void turbulence( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
void mountain_waves( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
void thermal( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
void collision( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
void hydro_const_theta( real z , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  real       p,exner,rt;
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
void hydro_const_bvfreq( real z , real bv_freq0 , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  real       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad ) {
  real dist;
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
void output( real3d &state , real etime ) {
  int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid, dimids[3];
  int i, k;
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];
  //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
  doub2d dens, uwnd, wwnd, theta;
  double etimearr[1];
  //Inform the user
  if (masterproc) { printf("*** OUTPUT ***\n"); }
  //Allocate some (big) temp arrays
  dens     = doub2d( "dens"     , nz,nx );
  uwnd     = doub2d( "uwnd"     , nz,nx );
  wwnd     = doub2d( "wwnd"     , nz,nx );
  theta    = doub2d( "theta"    , nz,nx );

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
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (k=0; k<nz; k++) {
    for (i=0; i<nx; i++) {
      dens (k,i) = state(ID_DENS,hs+k,hs+i);
      uwnd (k,i) = state(ID_UMOM,hs+k,hs+i) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) );
      wwnd (k,i) = state(ID_WMOM,hs+k,hs+i) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) );
      theta(k,i) = ( state(ID_RHOT,hs+k,hs+i) + hy_dens_theta_cell(hs+k) ) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) ) - hy_dens_theta_cell(hs+k) / hy_dens_cell(hs+k);
    }
  }

  //Write the grid data to file with all the processes writing collectively
  st3[0] = num_out; st3[1] = k_beg; st3[2] = i_beg;
  ct3[0] = 1      ; ct3[1] = nz   ; ct3[2] = nx   ;
  ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens.data()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd.data()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd.data()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta.data() ) , __LINE__ );

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
  ierr = MPI_Finalize();
  hy_dens_cell      .deallocate();
  hy_dens_theta_cell.deallocate();
  hy_dens_int       .deallocate();
  hy_dens_theta_int .deallocate();
  hy_pressure_int   .deallocate();
  state             .deallocate();
  state_tmp         .deallocate();
  flux              .deallocate();
  tend              .deallocate();
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
void reductions( double &mass , double &te ) {
  mass = 0;
  te   = 0;
  for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      double r  =   state(ID_DENS,hs+k,hs+i) + hy_dens_cell(hs+k);             // Density
      double u  =   state(ID_UMOM,hs+k,hs+i) / r;                              // U-wind
      double w  =   state(ID_WMOM,hs+k,hs+i) / r;                              // W-wind
      double th = ( state(ID_RHOT,hs+k,hs+i) + hy_dens_theta_cell(hs+k) ) / r; // Potential Temperature (theta)
      double p  = C0*pow(r*th,gamm);                               // Pressure
      double t  = th / pow(p0/p,rd/cp);                            // Temperature
      double ke = r*(u*u+w*w);                                     // Kinetic Energy
      double ie = r*cv*t;                                          // Internal Energy
      mass += r        *dx*dz; // Accumulate domain mass
      te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
  double glob[2], loc[2];
  loc[0] = mass;
  loc[1] = te;
  int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  mass = glob[0];
  te   = glob[1];
}


