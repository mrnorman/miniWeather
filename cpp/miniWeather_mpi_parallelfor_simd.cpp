
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
#include "const.h"
#include "pnetcdf.h"
#include <ctime>
#include <chrono>

// We're going to define all arrays on the host because this doesn't use parallel_for
typedef yakl::Array<real  ,1,yakl::memDevice> real1d;
typedef yakl::Array<real  ,2,yakl::memDevice> real2d;
typedef yakl::Array<real  ,3,yakl::memDevice> real3d;
typedef yakl::Array<double,1,yakl::memDevice> doub1d;
typedef yakl::Array<double,2,yakl::memDevice> doub2d;
typedef yakl::Array<double,3,yakl::memDevice> doub3d;

typedef yakl::Array<real   const,1,yakl::memDevice> realConst1d;
typedef yakl::Array<real   const,2,yakl::memDevice> realConst2d;
typedef yakl::Array<real   const,3,yakl::memDevice> realConst3d;
typedef yakl::Array<double const,1,yakl::memDevice> doubConst1d;
typedef yakl::Array<double const,2,yakl::memDevice> doubConst2d;
typedef yakl::Array<double const,3,yakl::memDevice> doubConst3d;

// Some arrays still need to be on the host, so we will explicitly create Host Array typedefs
typedef yakl::Array<real  ,1,yakl::memHost> real1dHost;
typedef yakl::Array<real  ,2,yakl::memHost> real2dHost;
typedef yakl::Array<real  ,3,yakl::memHost> real3dHost;
typedef yakl::Array<double,1,yakl::memHost> doub1dHost;
typedef yakl::Array<double,2,yakl::memHost> doub2dHost;
typedef yakl::Array<double,3,yakl::memHost> doub3dHost;

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
struct Fixed_data {
  int nx, nz;                 //Number of local grid cells in the x- and z- dimensions for this MPI task
  int i_beg, k_beg;           //beginning index in the x- and z-directions for this MPI task
  int nranks, myrank;         //Number of MPI ranks and my rank id
  int left_rank, right_rank;  //MPI Rank IDs that exist to my left and right in the global domain
  int masterproc;             //Am I the master process (rank == 0)?
  realConst1d hy_dens_cell;        //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
  realConst1d hy_dens_theta_cell;  //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
  realConst1d hy_dens_int;         //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
  realConst1d hy_dens_theta_int;   //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
  realConst1d hy_pressure_int;     //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)
};

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////

//Declaring the functions defined after "main"
void init                 ( real3d &state , real &dt , Fixed_data &fixed_data );
void finalize             ( );
YAKL_INLINE void injection            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void density_current      ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void gravity_waves        ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void thermal              ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void collision            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void hydro_const_theta    ( real z                    , real &r , real &t );
YAKL_INLINE void hydro_const_bvfreq   ( real z , real bv_freq0    , real &r , real &t );
YAKL_INLINE real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad );
void output               ( realConst3d state , real etime , int &num_out , Fixed_data const &fixed_data );
void ncwrap               ( int ierr , int line );
void perform_timestep     ( real3d const &state , real dt , int &direction_switch , Fixed_data const &fixed_data );
void semi_discrete_step   ( realConst3d state_init , real3d const &state_forcing , real3d const &state_out , real dt ,
                            int dir , Fixed_data const &fixed_data );
void compute_tendencies_x ( realConst3d state , real3d const &tend , real dt , Fixed_data const &fixed_data );
void compute_tendencies_z ( realConst3d state , real3d const &tend , real dt , Fixed_data const &fixed_data );
void set_halo_values_x    ( real3d const &state  , Fixed_data const &fixed_data );
void set_halo_values_z    ( real3d const &state  , Fixed_data const &fixed_data );
void reductions           ( realConst3d state , double &mass , double &te , Fixed_data const &fixed_data );


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  yakl::init();
  {
    Fixed_data fixed_data;
    real3d state;
    real dt;                    //Model time step (seconds)

    // Init allocates the state and hydrostatic arrays hy_*
    init( state , dt , fixed_data );

    auto &masterproc = fixed_data.masterproc;

    //Initial reductions for mass, kinetic energy, and total energy
    real mass0, te0;
    reductions(state,mass0,te0,fixed_data);

    int  num_out = 0;          //The number of outputs performed so far
    real output_counter = 0;   //Helps determine when it's time to do output
    real etime = 0;

    //Output the initial state
    output(state,etime,num_out,fixed_data);

    int direction_switch = 1;  // Tells dimensionally split which order to take x,z solves

    ////////////////////////////////////////////////////
    // MAIN TIME STEP LOOP
    ////////////////////////////////////////////////////
    auto t1 = std::chrono::steady_clock::now();
    while (etime < sim_time) {
      //If the time step leads to exceeding the simulation time, shorten it for the last step
      if (etime + dt > sim_time) { dt = sim_time - etime; }
      //Perform a single time step
      perform_timestep(state,dt,direction_switch,fixed_data);
      //Inform the user
      #ifndef NO_INFORM
        if (masterproc) { printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); }
      #endif
      //Update the elapsed time and output counter
      etime = etime + dt;
      output_counter = output_counter + dt;
      //If it's time for output, reset the counter, and do output
      if (output_counter >= output_freq) {
        output_counter = output_counter - output_freq;
        output(state,etime,num_out,fixed_data);
      }
    }
    auto t2 = std::chrono::steady_clock::now();
    if (masterproc) {
      std::cout << "CPU Time: " << std::chrono::duration<double>(t2-t1).count() << " sec\n";
    }

    //Final reductions for mass, kinetic energy, and total energy
    real mass, te;
    reductions(state,mass,te,fixed_data);

    if (masterproc) {
      printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
      printf( "d_te:   %le\n" , (te   - te0  )/te0   );
    }

    finalize();
  }
  yakl::finalize();
  MPI_Finalize();
}


//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q_n + dt/3 * rhs(q_n)
// q**    = q_n + dt/2 * rhs(q* )
// q_n+1  = q_n + dt/1 * rhs(q**)
void perform_timestep( real3d const &state , real dt , int &direction_switch , Fixed_data const &fixed_data) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;

  real3d state_tmp("state_tmp",NUM_VARS,nz+2*hs,nx+2*hs);

  if (direction_switch) {
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , fixed_data );
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , fixed_data );
  } else {
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , fixed_data );
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , fixed_data );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}


//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step( realConst3d state_init , real3d const &state_forcing , real3d const &state_out , real dt , int dir , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &i_beg              = fixed_data.i_beg             ;
  auto &k_beg              = fixed_data.k_beg             ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;

  real3d tend("tend",NUM_VARS,nz,nx);

  if        (dir == DIR_X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    set_halo_values_x(state_forcing,fixed_data);
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(state_forcing,tend,dt,fixed_data);
  } else if (dir == DIR_Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    set_halo_values_z(state_forcing,fixed_data);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(state_forcing,tend,dt,fixed_data);
  }

  //Apply the tendencies to the fluid state
  // for (ll=0; ll<NUM_VARS; ll++) {
  //   for (k=0; k<nz; k++) {
  //     for (i=0; i<nx; i++) {
  parallel_for( SimpleBounds<3>(NUM_VARS,nz,nx) , YAKL_LAMBDA ( int ll, int k, int i ) {
    if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
      real x = (i_beg + i+0.5)*dx;
      real z = (k_beg + k+0.5)*dz;
      real wpert = sample_ellipse_cosine( x,z , 0.01 , xlen/8,1000., 500.,500. );
      tend(ID_WMOM,k,i) += wpert*hy_dens_cell(hs+k);
    }
    state_out(ll,hs+k,hs+i) = state_init(ll,hs+k,hs+i) + dt * tend(ll,k,i);
  });
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x( realConst3d state , real3d const &tend , real dt , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  real3d flux("flux",NUM_VARS,nz,nx+1);

  //Compute fluxes in the x-direction for each cell
  // for (k=0; k<nz; k++) {
  //   for (i=0; i<nx+1; i++) {
  int xdim = nx+1;
  int xblocks = (xdim-1)/simd_len + 1;
  parallel_for( SimpleBounds<2>(nz,xblocks) , YAKL_LAMBDA (int k, int iblk) {
    SArray<Pack<real,simd_len>,1,4> stencil;
    SArray<Pack<real,simd_len>,1,NUM_VARS> d3_vals;
    SArray<Pack<real,simd_len>,1,NUM_VARS> vals;
    //Compute the hyperviscosity coeficient
    real hv_coef = -hv_beta * dx / (16*dt);

    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (int ll=0; ll<NUM_VARS; ll++) {
      for (int s=0; s < sten_size; s++) {
        for (int ilane=0; ilane < simd_len; ilane++) {
          int i = min( xdim-1 , iblk*simd_len + ilane );
          stencil(s)(ilane) = state(ll,hs+k,i+s);
        }
      }
      //Fourth-order-accurate interpolation of the state
      vals(ll) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
      //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
      d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    Pack<real,simd_len> r = vals(ID_DENS) + hy_dens_cell(hs+k);
    Pack<real,simd_len> u = vals(ID_UMOM) / r;
    Pack<real,simd_len> w = vals(ID_WMOM) / r;
    Pack<real,simd_len> t = ( vals(ID_RHOT) + hy_dens_theta_cell(hs+k) ) / r;
    Pack<real,simd_len> p = C0*pow((r*t),gamm);

    Pack<real,simd_len> f1 = r*u     - hv_coef*d3_vals(ID_DENS);
    Pack<real,simd_len> f2 = r*u*u+p - hv_coef*d3_vals(ID_UMOM);
    Pack<real,simd_len> f3 = r*u*w   - hv_coef*d3_vals(ID_WMOM);
    Pack<real,simd_len> f4 = r*u*t   - hv_coef*d3_vals(ID_RHOT);

    //Compute the flux vector
    for (int ilane=0; ilane < simd_len; ilane++) {
      int i = min(xdim-1 , iblk*simd_len + ilane);
      flux(ID_DENS,k,i) = f1(ilane);
      flux(ID_UMOM,k,i) = f2(ilane);
      flux(ID_WMOM,k,i) = f3(ilane);
      flux(ID_RHOT,k,i) = f4(ilane);
    }
  });

  //Use the fluxes to compute tendencies for each cell
  // for (ll=0; ll<NUM_VARS; ll++) {
  //   for (k=0; k<nz; k++) {
  //     for (i=0; i<nx; i++) {
  parallel_for( SimpleBounds<3>(NUM_VARS,nz,nx) , YAKL_LAMBDA ( int ll, int k, int i ) {
    tend(ll,k,i) = -( flux(ll,k,i+1) - flux(ll,k,i) ) / dx;
  });
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z( realConst3d state , real3d const &tend , real dt , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_int        = fixed_data.hy_dens_int       ;
  auto &hy_dens_theta_int  = fixed_data.hy_dens_theta_int ;
  auto &hy_pressure_int    = fixed_data.hy_pressure_int   ;

  real3d flux("flux",NUM_VARS,nz+1,nx);

  //Compute fluxes in the x-direction for each cell
  // for (k=0; k<nz+1; k++) {
  //   for (i=0; i<nx; i++) {
  int xdim = nx+1;
  int xblocks = (xdim-1)/simd_len + 1;
  parallel_for( SimpleBounds<2>(nz+1,xblocks) , YAKL_LAMBDA (int k, int iblk) {
    SArray<Pack<real,simd_len>,1,4> stencil;
    SArray<Pack<real,simd_len>,1,NUM_VARS> d3_vals;
    SArray<Pack<real,simd_len>,1,NUM_VARS> vals;
    //Compute the hyperviscosity coeficient
    real hv_coef = -hv_beta * dz / (16*dt);

    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (int ll=0; ll<NUM_VARS; ll++) {
      for (int s=0; s<sten_size; s++) {
        for (int ilane = 0; ilane < simd_len; ilane++) {
          int i = min( xdim-1 , iblk*simd_len + ilane );
          stencil(s)(ilane) = state(ll,k+s,hs+i);
        }
      }
      //Fourth-order-accurate interpolation of the state
      vals(ll) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
      //First-order-accurate interpolation of the third spatial derivative of the state
      d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    Pack<real,simd_len> r = vals(ID_DENS) + hy_dens_int(k);
    Pack<real,simd_len> u = vals(ID_UMOM) / r;
    Pack<real,simd_len> w = vals(ID_WMOM) / r;
    Pack<real,simd_len> t = ( vals(ID_RHOT) + hy_dens_theta_int(k) ) / r;
    Pack<real,simd_len> p = C0*pow((r*t),gamm) - hy_pressure_int(k);
    if (k == 0 || k == nz) {
      w                = 0;
      d3_vals(ID_DENS) = 0;
    }

    Pack<real,simd_len> f1 = r*w     - hv_coef*d3_vals(ID_DENS);
    Pack<real,simd_len> f2 = r*w*u   - hv_coef*d3_vals(ID_UMOM);
    Pack<real,simd_len> f3 = r*w*w+p - hv_coef*d3_vals(ID_WMOM);
    Pack<real,simd_len> f4 = r*w*t   - hv_coef*d3_vals(ID_RHOT);

    //Compute the flux vector with hyperviscosity
    for (int ilane = 0; ilane < simd_len; ilane++) {
      int i = min( xdim-1 , iblk*simd_len + ilane );
      flux(ID_DENS,k,i) = f1(ilane);
      flux(ID_UMOM,k,i) = f2(ilane);
      flux(ID_WMOM,k,i) = f3(ilane);
      flux(ID_RHOT,k,i) = f4(ilane);
    }
  });

  //Use the fluxes to compute tendencies for each cell
  // for (ll=0; ll<NUM_VARS; ll++) {
  //   for (k=0; k<nz; k++) {
  //     for (i=0; i<nx; i++) {
  parallel_for( SimpleBounds<3>(NUM_VARS,nz,nx) , YAKL_LAMBDA ( int ll, int k, int i ) {
    tend(ll,k,i) = -( flux(ll,k+1,i) - flux(ll,k,i) ) / dz;
    if (ll == ID_WMOM) {
      tend(ll,k,i) -= state(ID_DENS,hs+k,hs+i)*grav;
    }
  });
}



//Set this MPI task's halo values in the x-direction. This routine will require MPI
void set_halo_values_x( real3d const &state , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &k_beg              = fixed_data.k_beg             ;
  auto &left_rank          = fixed_data.left_rank         ;
  auto &right_rank         = fixed_data.right_rank        ;
  auto &myrank             = fixed_data.myrank            ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  int ierr;
  MPI_Request req_r[2], req_s[2];
  MPI_Datatype type;

  if (std::is_same<real,float>::value) {
    type = MPI_FLOAT;
  } else {
    type = MPI_DOUBLE;
  }

  real3d     sendbuf_l    ( "sendbuf_l" , NUM_VARS,nz,hs );  //Buffer to send data to the left MPI rank
  real3d     sendbuf_r    ( "sendbuf_r" , NUM_VARS,nz,hs );  //Buffer to send data to the right MPI rank
  real3d     recvbuf_l    ( "recvbuf_l" , NUM_VARS,nz,hs );  //Buffer to receive data from the left MPI rank
  real3d     recvbuf_r    ( "recvbuf_r" , NUM_VARS,nz,hs );  //Buffer to receive data from the right MPI rank
  real3dHost sendbuf_l_cpu( "sendbuf_l" , NUM_VARS,nz,hs );  //Buffer to send data to the left MPI rank (CPU copy)
  real3dHost sendbuf_r_cpu( "sendbuf_r" , NUM_VARS,nz,hs );  //Buffer to send data to the right MPI rank (CPU copy)
  real3dHost recvbuf_l_cpu( "recvbuf_l" , NUM_VARS,nz,hs );  //Buffer to receive data from the left MPI rank (CPU copy)
  real3dHost recvbuf_r_cpu( "recvbuf_r" , NUM_VARS,nz,hs );  //Buffer to receive data from the right MPI rank (CPU copy)

  //Prepost receives
  ierr = MPI_Irecv(recvbuf_l_cpu.data(),hs*nz*NUM_VARS,type, left_rank,0,MPI_COMM_WORLD,&req_r[0]);
  ierr = MPI_Irecv(recvbuf_r_cpu.data(),hs*nz*NUM_VARS,type,right_rank,1,MPI_COMM_WORLD,&req_r[1]);

  //Pack the send buffers
  // for (ll=0; ll<NUM_VARS; ll++) {
  //   for (k=0; k<nz; k++) {
  //     for (s=0; s<hs; s++) {
  parallel_for( SimpleBounds<3>(NUM_VARS,nz,hs) , YAKL_LAMBDA (int ll, int k, int s) {
    sendbuf_l(ll,k,s) = state(ll,k+hs,hs+s);
    sendbuf_r(ll,k,s) = state(ll,k+hs,nx+s);
  });
  yakl::fence();

  // This will copy from GPU to host
  sendbuf_l.deep_copy_to(sendbuf_l_cpu);
  sendbuf_r.deep_copy_to(sendbuf_r_cpu);
  yakl::fence();

  //Fire off the sends
  ierr = MPI_Isend(sendbuf_l_cpu.data(),hs*nz*NUM_VARS,type, left_rank,1,MPI_COMM_WORLD,&req_s[0]);
  ierr = MPI_Isend(sendbuf_r_cpu.data(),hs*nz*NUM_VARS,type,right_rank,0,MPI_COMM_WORLD,&req_s[1]);

  //Wait for receives to finish
  ierr = MPI_Waitall(2,req_r,MPI_STATUSES_IGNORE);

  // This will copy from host to GPU
  recvbuf_l_cpu.deep_copy_to(recvbuf_l);
  recvbuf_r_cpu.deep_copy_to(recvbuf_r);
  yakl::fence();

  //Unpack the receive buffers
  // for (ll=0; ll<NUM_VARS; ll++) {
  //   for (k=0; k<nz; k++) {
  //     for (s=0; s<hs; s++) {
  parallel_for( SimpleBounds<3>(NUM_VARS,nz,hs) , YAKL_LAMBDA (int ll, int k, int s) {
    state(ll,k+hs,s      ) = recvbuf_l(ll,k,s);
    state(ll,k+hs,nx+hs+s) = recvbuf_r(ll,k,s);
  });
  yakl::fence();

  //Wait for sends to finish
  ierr = MPI_Waitall(2,req_s,MPI_STATUSES_IGNORE);

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      // for (k=0; k<nz; k++) {
      //   for (i=0; i<hs; i++) {
      parallel_for( SimpleBounds<2>(nz,hs) , YAKL_LAMBDA (int k, int i) {
        double z = (k_beg + k+0.5)*dz;
        if (abs(z-3*zlen/4) <= zlen/16) {
          state(ID_UMOM,hs+k,i) = (state(ID_DENS,hs+k,i)+hy_dens_cell(hs+k)) * 50.;
          state(ID_RHOT,hs+k,i) = (state(ID_DENS,hs+k,i)+hy_dens_cell(hs+k)) * 298. - hy_dens_theta_cell(hs+k);
        }
      });
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void set_halo_values_z( real3d const &state , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  
  // for (ll=0; ll<NUM_VARS; ll++) {
  //   for (i=0; i<nx+2*hs; i++) {
  parallel_for( SimpleBounds<2>(NUM_VARS,nx+2*hs) , YAKL_LAMBDA (int ll, int i) {
    if (ll == ID_WMOM) {
      state(ll,0      ,i) = 0.;
      state(ll,1      ,i) = 0.;
      state(ll,nz+hs  ,i) = 0.;
      state(ll,nz+hs+1,i) = 0.;
    } else if (ll == ID_UMOM) {
      state(ll,0      ,i) = state(ll,hs     ,i) / hy_dens_cell(hs     ) * hy_dens_cell(0      );
      state(ll,1      ,i) = state(ll,hs     ,i) / hy_dens_cell(hs     ) * hy_dens_cell(1      );
      state(ll,nz+hs  ,i) = state(ll,nz+hs-1,i) / hy_dens_cell(nz+hs-1) * hy_dens_cell(nz+hs  );
      state(ll,nz+hs+1,i) = state(ll,nz+hs-1,i) / hy_dens_cell(nz+hs-1) * hy_dens_cell(nz+hs+1);
    } else {
      state(ll,0      ,i) = state(ll,hs     ,i);
      state(ll,1      ,i) = state(ll,hs     ,i);
      state(ll,nz+hs  ,i) = state(ll,nz+hs-1,i);
      state(ll,nz+hs+1,i) = state(ll,nz+hs-1,i);
    }
  });
}


void init( real3d &state , real &dt , Fixed_data &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &i_beg              = fixed_data.i_beg             ;
  auto &k_beg              = fixed_data.k_beg             ;
  auto &left_rank          = fixed_data.left_rank         ;
  auto &right_rank         = fixed_data.right_rank        ;
  auto &nranks             = fixed_data.nranks            ;
  auto &myrank             = fixed_data.myrank            ;
  auto &masterproc         = fixed_data.masterproc        ;
  int  ierr;

  ierr = MPI_Comm_size(MPI_COMM_WORLD,&nranks);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  real nper = ( (double) nx_glob ) / nranks;
  i_beg = round( nper* (myrank)    );
  int i_end = round( nper*((myrank)+1) )-1;
  nx = i_end - i_beg + 1;
  left_rank  = myrank - 1;
  if (left_rank == -1) left_rank = nranks-1;
  right_rank = myrank + 1;
  if (right_rank == nranks) right_rank = 0;

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  k_beg = 0;
  nz = nz_glob;
  masterproc = (myrank == 0);

  //Allocate the model data
  state              = real3d( "state" , NUM_VARS,nz+2*hs,nx+2*hs);

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = min(dx,dz) / max_speed * cfl;

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
  // for (k=0; k<nz+2*hs; k++) {
  //   for (i=0; i<nx+2*hs; i++) {
  parallel_for( SimpleBounds<2>(nz+2*hs,nx+2*hs) , YAKL_LAMBDA (int k, int i) {
    //Initialize the state to zero
    for (int ll=0; ll<NUM_VARS; ll++) {
      state(ll,k,i) = 0.;
    }
    //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
    for (int kk=0; kk<nqpoints; kk++) {
      for (int ii=0; ii<nqpoints; ii++) {
        //Compute the x,z location within the global domain based on cell and quadrature index
        real x = (i_beg + i-hs+0.5)*dx + (qpoints(ii)-0.5)*dx;
        real z = (k_beg + k-hs+0.5)*dz + (qpoints(kk)-0.5)*dz;
        real r, u, w, t, hr, ht;

        //Set the fluid state based on the user's specification
        if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
        if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
        if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (x,z,r,u,w,t,hr,ht); }
        if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
        if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

        //Store into the fluid state array
        state(ID_DENS,k,i) += r                         * qweights(ii)*qweights(kk);
        state(ID_UMOM,k,i) += (r+hr)*u                  * qweights(ii)*qweights(kk);
        state(ID_WMOM,k,i) += (r+hr)*w                  * qweights(ii)*qweights(kk);
        state(ID_RHOT,k,i) += ( (r+hr)*(t+ht) - hr*ht ) * qweights(ii)*qweights(kk);
      }
    }
  });

  real1d hy_dens_cell      ("hy_dens_cell      ",nz+2*hs);
  real1d hy_dens_theta_cell("hy_dens_theta_cell",nz+2*hs);
  real1d hy_dens_int       ("hy_dens_int       ",nz+1);
  real1d hy_dens_theta_int ("hy_dens_theta_int ",nz+1);
  real1d hy_pressure_int   ("hy_pressure_int   ",nz+1);

  //Compute the hydrostatic background state over vertical cell averages
  // for (int k=0; k<nz+2*hs; k++) {
  parallel_for( nz+2*hs , YAKL_LAMBDA (int k) {
    hy_dens_cell      (k) = 0.;
    hy_dens_theta_cell(k) = 0.;
    for (int kk=0; kk<nqpoints; kk++) {
      real z = (k_beg + k-hs+0.5)*dz;
      real r, u, w, t, hr, ht;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      (k) = hy_dens_cell      (k) + hr    * qweights(kk);
      hy_dens_theta_cell(k) = hy_dens_theta_cell(k) + hr*ht * qweights(kk);
    }
  });
  //Compute the hydrostatic background state at vertical cell interfaces
  // for (int k=0; k<nz+1; k++) {
  parallel_for( nz+1 , YAKL_LAMBDA (int k) {
    real z = (k_beg + k)*dz;
    real r, u, w, t, hr, ht;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      (k) = hr;
    hy_dens_theta_int(k) = hr*ht;
    hy_pressure_int  (k) = C0*pow((hr*ht),gamm);
  });

  fixed_data.hy_dens_cell       = realConst1d(hy_dens_cell      );
  fixed_data.hy_dens_theta_cell = realConst1d(hy_dens_theta_cell);
  fixed_data.hy_dens_int        = realConst1d(hy_dens_int       );
  fixed_data.hy_dens_theta_int  = realConst1d(hy_dens_theta_int );
  fixed_data.hy_pressure_int    = realConst1d(hy_pressure_int   );
}


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
YAKL_INLINE void injection( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
YAKL_INLINE void density_current( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
YAKL_INLINE void gravity_waves ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
YAKL_INLINE void thermal( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
YAKL_INLINE void collision( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
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
YAKL_INLINE void hydro_const_theta( real z , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  real exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  real p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  real rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrstatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
YAKL_INLINE void hydro_const_bvfreq( real z , real bv_freq0 , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  real exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  real p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  real rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
YAKL_INLINE real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad ) {
  //Compute distance from bubble center
  real dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
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
void output( realConst3d state , real etime , int &num_out , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &i_beg              = fixed_data.i_beg             ;
  auto &k_beg              = fixed_data.k_beg             ;
  auto &masterproc         = fixed_data.masterproc        ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid, dimids[3];
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];
  //Inform the user
  if (masterproc) { printf("*** OUTPUT ***\n"); }
  //Allocate some (big) temp arrays
  doub2d dens ( "dens"     , nz,nx );
  doub2d uwnd ( "uwnd"     , nz,nx );
  doub2d wwnd ( "wwnd"     , nz,nx );
  doub2d theta( "theta"    , nz,nx );

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
  // for (k=0; k<nz; k++) {
  //   for (i=0; i<nx; i++) {
  parallel_for( SimpleBounds<2>(nz,nx) , YAKL_LAMBDA (int k, int i) {
    dens (k,i) = state(ID_DENS,hs+k,hs+i);
    uwnd (k,i) = state(ID_UMOM,hs+k,hs+i) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) );
    wwnd (k,i) = state(ID_WMOM,hs+k,hs+i) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) );
    theta(k,i) = ( state(ID_RHOT,hs+k,hs+i) + hy_dens_theta_cell(hs+k) ) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) ) - hy_dens_theta_cell(hs+k) / hy_dens_cell(hs+k);
  });
  yakl::fence();

  //Write the grid data to file with all the processes writing collectively
  st3[0] = num_out; st3[1] = k_beg; st3[2] = i_beg;
  ct3[0] = 1      ; ct3[1] = nz   ; ct3[2] = nx   ;
  ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens .createHostCopy().data() ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd .createHostCopy().data() ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd .createHostCopy().data() ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta.createHostCopy().data() ) , __LINE__ );

  //Only the master process needs to write the elapsed time
  //Begin "independent" write mode
  ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
  //write elapsed time to file
  if (masterproc) {
    st1[0] = num_out;
    ct1[0] = 1;
    double etimearr[1];
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
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
void reductions( realConst3d state, double &mass , double &te , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  doub2d mass2d("mass2d",nz,nx);
  doub2d te2d  ("te2d  ",nz,nx);

  // for (k=0; k<nz; k++) {
  //   for (i=0; i<nx; i++) {
  parallel_for( SimpleBounds<2>(nz,nx) , YAKL_LAMBDA (int k, int i) {
    double r  =   state(ID_DENS,hs+k,hs+i) + hy_dens_cell(hs+k);             // Density
    double u  =   state(ID_UMOM,hs+k,hs+i) / r;                              // U-wind
    double w  =   state(ID_WMOM,hs+k,hs+i) / r;                              // W-wind
    double th = ( state(ID_RHOT,hs+k,hs+i) + hy_dens_theta_cell(hs+k) ) / r; // Potential Temperature (theta)
    double p  = C0*pow(r*th,gamm);                               // Pressure
    double t  = th / pow(p0/p,rd/cp);                            // Temperature
    double ke = r*(u*u+w*w);                                     // Kinetic Energy
    double ie = r*cv*t;                                          // Internal Energy
    mass2d(k,i) = r        *dx*dz; // Accumulate domain mass
    te2d  (k,i) = (ke + ie)*dx*dz; // Accumulate domain total energy
  });
  mass = yakl::intrinsics::sum( mass2d );
  te   = yakl::intrinsics::sum( te2d   );

  double glob[2], loc[2];
  loc[0] = mass;
  loc[1] = te;
  int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  mass = glob[0];
  te   = glob[1];
}


