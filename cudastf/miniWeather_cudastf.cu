//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//////////////////////////////////////////////////////////////////////////////////////////

/*
** Copyright (c) 2018, National Center for Computational Sciences, Oak Ridge National Laboratory. All rights reserved.
**
** Portions Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
*/

/**
 * @file
 * @brief CUDASTF implementation of the ORNL's miniWeather CFD code
 */

#include <cuda/experimental/stf.cuh>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef HAVE_NETCDF
#    include <netcdf.h>
#endif

using namespace cuda::experimental::stf;

using policy = blocked_partition_custom<1>;

struct state_t {
    state_t(context& ctx, size_t nx, size_t nz, size_t hs, size_t NUM_VARS) {
        l = ctx.logical_data<double>(nx + 2 * hs, nz + 2 * hs, NUM_VARS);
    }

    // double *vals;
    logical_data<slice<double, 3>> l;
};

struct tend_t {
    tend_t(context& ctx, size_t nx, size_t nz, size_t NUM_VARS) {
        l = ctx.logical_data<double>(nx, nz, NUM_VARS);
        l.set_symbol("tend");
    }

    logical_data<slice<double, 3>> l;
};

const double grav = 9.8;                              // Gravitational acceleration (m / s^2)
const double cp = 1004.;                              // Specific heat of dry air at constant pressure
const double rd = 287.;                               // Dry air constant for equation of state (P=rho*rd*T)
const double p0 = 1.e5;                               // Standard pressure at the surface in Pascals
const double C0 = 27.5629410929725921310572974482;    // Constant to translate potential temperature into pressure
                                                      // (P=C0*(rho*theta)**gamma)
const double gamm = 1.40027894002789400278940027894;  // gamma=cp/Rd , have to call this gamm because "gamma" is taken
                                                      // (I hate C so much)
// Define domain and stability-related constants
const double xlen = 2.e4;     // Length of the domain in the x-direction (meters)
const double zlen = 1.e4;     // Length of the domain in the z-direction (meters)
const double hv_beta = 0.25;  // How strong to diffuse the solution: hv_beta \in [0:1]
const double cfl = 1.50;      //"Courant, Friedrichs, Lewy" number (for numerical stability)
const double max_speed =
        450;       // Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs = 2;  //"Halo" size: number of cells needed for a full "stencil" of information for reconstruction
const int sten_size = 4;  // Size of the stencil used for interpolation

// Parameters for indexing and flags
const int NUM_VARS = 4;  // Number of fluid state variables
const int ID_DENS = 0;   // index for density ("rho")
const int ID_UMOM = 1;   // index for momentum in the x-direction ("rho * u")
const int ID_WMOM = 2;   // index for momentum in the z-direction ("rho * w")
const int ID_RHOT = 3;   // index for density * potential temperature ("rho * theta")
const int DIR_X = 1;     // Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;     // Integer constant to express that this operation is in the z-direction

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double sim_time;            // total simulation time in seconds
double output_freq;         // frequency to perform output in seconds
double dt;                  // Model time step (seconds)
int nx, nz;                 // Number of local grid cells in the x- and z- dimensions
double dx, dz;              // Grid space length in x- and z-dimension (meters)
int nx_glob, nz_glob;       // Number of total grid cells in the x- and z- dimensions
int i_beg, k_beg;           // beginning index in the x- and z-directions
int nranks, myrank;         // my rank id
int left_rank, right_rank;  // Rank IDs that exist to my left and right in the global domain

struct boundaries_t {
    boundaries_t(context& ctx, size_t nz, size_t hs) {
        lhy_dens_cell = ctx.logical_data<double>(nz + 2 + hs);
        lhy_dens_theta_cell = ctx.logical_data<double>(nz + 2 + hs);
        lhy_dens_int = ctx.logical_data<double>(nz + 1);
        lhy_dens_theta_int = ctx.logical_data<double>(nz + 1);
        lhy_pressure_int = ctx.logical_data<double>(nz + 1);

        lhy_dens_cell.set_symbol("hy_dens_cell");
        lhy_dens_theta_cell.set_symbol("hy_dens_theta_cell");
        lhy_dens_int.set_symbol("hy_dens_int");
        lhy_dens_theta_int.set_symbol("hy_dens_theta_int");
        lhy_pressure_int.set_symbol("hy_pressure_int");
    }

    logical_data<slice<double>> lhy_dens_cell;  // hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
    logical_data<slice<double>> lhy_dens_theta_cell;  // hydrostatic rho*t (vert cell avgs).     Dimensions:
                                                      // (1-hs:nz+hs)
    logical_data<slice<double>> lhy_dens_int;         // hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
    logical_data<slice<double>> lhy_dens_theta_int;   // hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
    logical_data<slice<double>> lhy_pressure_int;     // hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

    frozen_logical_data<slice<double>> flhy_dens_cell;
    frozen_logical_data<slice<double>> flhy_dens_theta_cell;
    frozen_logical_data<slice<double>> flhy_dens_int;
    frozen_logical_data<slice<double>> flhy_dens_theta_int;
    frozen_logical_data<slice<double>> flhy_pressure_int;

    slice<double> hy_dens_cell;
    slice<double> hy_dens_theta_cell;
    slice<double> hy_dens_int;
    slice<double> hy_dens_theta_int;
    slice<double> hy_pressure_int;
};

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;           // Elapsed model time
double output_counter;  // Helps determine when it's time to do output
int num_out = 0;        // The number of outputs performed so far
int direction_switch = 1;

// Declaring the functions defined after "main"
void init(exec_place& where, context& ctx, state_t& state, state_t& state_tmp, boundaries_t& b);

__host__ __device__ void injection(
        double x, double z, double& r, double& u, double& w, double& t, double& hr, double& ht);
__host__ __device__ void hydro_const_theta(double z, double& r, double& t);

void perform_timestep(exec_place& where, context& ctx, state_t& state, state_t& state_tmp, boundaries_t& b, double dt);

void semi_discrete_step(exec_place& where, context& ctx, state_t& state_init, state_t& state_forcing,
        state_t& state_out, boundaries_t& b, double dt, int dir);

void compute_tendencies_x(exec_place& where, context& ctx, state_t& state, tend_t& tend, boundaries_t& b);

void compute_tendencies_z(exec_place& where, context& ctx, state_t& state, tend_t& tend, boundaries_t& b);

void set_halo_values_x(exec_place& where, context& ctx, state_t& state, boundaries_t& b);

void set_halo_values_z(exec_place& where, context& ctx, state_t& state);

#ifdef HAVE_NETCDF
void output(context& ctx, state_t& state, boundaries_t& b, double etime);
void ncwrap(int ierr, int line);
#endif  // HAVE_NETCDF

void simulation(context& ctx, exec_place where) {
    // printf("Using ctx %s on %s\n", ctx.to_string().c_str(), where.to_string().c_str());

    double exe_time;
    struct timeval stop_time, start_time;

    // Runtime variable arrays
    state_t state(ctx, nx_glob, nz_glob, hs,
            NUM_VARS);  // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    state_t state_tmp(ctx, nx_glob, nz_glob, hs,
            NUM_VARS);  // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)

    state.l.set_symbol("state");
    state_tmp.l.set_symbol("state_tmp");

    boundaries_t b(ctx, nz_glob, hs);

    init(where, ctx, state, state_tmp, b);

#ifdef HAVE_NETCDF
    // Output the initial state
    output(ctx, state, b, etime);
#endif

    gettimeofday(&start_time, NULL);

    ctx.repeat([&]() { return etime < sim_time; })->*[&](context ctx, size_t) {
        // If the time step leads to exceeding the simulation time, shorten it for the last step
        if (etime + dt > sim_time) {
            dt = sim_time - etime;
        }

        // Perform a single time step
        perform_timestep(where, ctx, state, state_tmp, b, dt);

        // Inform the user
        fprintf(stderr, "Elapsed Time: %lf / %lf\n", etime, sim_time);

        // Update the elapsed time and output counter
        etime = etime + dt;

#ifdef HAVE_NETCDF
        output_counter = output_counter + dt;
        // If it's time for output, reset the counter, and do output

        if (output_counter >= output_freq) {
            output_counter = output_counter - output_freq;
            output(ctx, state, b, etime);
        }
#endif
    };

    ctx.finalize();

    gettimeofday(&stop_time, NULL);
    exe_time = (stop_time.tv_sec + (stop_time.tv_usec / 1000000.0)) -
               (start_time.tv_sec + (start_time.tv_usec / 1000000.0));

    printf("Complete. Execution time is = %lf seconds\n", exe_time);
}

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    ///////////////////////////////////////////////////////////////////////////////////////
    // BEGIN USER-CONFIGURABLE PARAMETERS
    ///////////////////////////////////////////////////////////////////////////////////////
    // The x-direction length is twice as long as the z-direction length
    // So, you'll want to have nx_glob be twice as large as nz_glob
    nx_glob = 400;     // Number of total cells in the x-dirction
    nz_glob = 200;     // Number of total cells in the z-dirction
    sim_time = 2;      // How many seconds to run the simulation
    output_freq = 10;  // How frequently to output data to file (in seconds)
    ///////////////////////////////////////////////////////////////////////////////////////
    // END USER-CONFIGURABLE PARAMETERS
    ///////////////////////////////////////////////////////////////////////////////////////

    if (argc >= 4) {
        fprintf(stdout, "The arguments supplied are %s %s %s\n", argv[1], argv[2], argv[3]);
        nx_glob = atoi(argv[1]);
        nz_glob = atoi(argv[2]);
        sim_time = atoi(argv[3]);
    } else {
        // printf("Using default values ...\n");
    }

    exec_place where = exec_place::current_device();

    if (argc >= 5) {
        int use_gpu = atoi(argv[4]);
        switch (use_gpu) {
        case 0: where = exec_place::host; break;
        case 1: where = exec_place::current_device(); break;
        case 2: where = exec_place::all_devices(); break;
        case 3: where = exec_place::repeat(exec_place::current_device(), 8); break;
        default: abort();
        }
    }

    context ctx;
    int use_graph = 0;
    if (argc >= 6) {
        use_graph = atoi(argv[5]);
        if (use_graph) {
            ctx = graph_ctx();
        }
    }
    // fprintf(stderr, "Using %s backend.\n", use_graph ? "graph" : "stream");

    simulation(ctx, where);
}

// Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
// The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
// order of directions is alternated each time step.
// The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void perform_timestep(exec_place& where, context& ctx, state_t& state, state_t& state_tmp, boundaries_t& b, double dt) {
    if (direction_switch) {
        // x-direction first
        semi_discrete_step(where, ctx, state, state, state_tmp, b, dt / 3, DIR_X);
        semi_discrete_step(where, ctx, state, state_tmp, state_tmp, b, dt / 2, DIR_X);
        semi_discrete_step(where, ctx, state, state_tmp, state, b, dt / 1, DIR_X);
        // z-direction second
        semi_discrete_step(where, ctx, state, state, state_tmp, b, dt / 3, DIR_Z);
        semi_discrete_step(where, ctx, state, state_tmp, state_tmp, b, dt / 2, DIR_Z);
        semi_discrete_step(where, ctx, state, state_tmp, state, b, dt / 1, DIR_Z);
    } else {
        // z-direction second
        semi_discrete_step(where, ctx, state, state, state_tmp, b, dt / 3, DIR_Z);
        semi_discrete_step(where, ctx, state, state_tmp, state_tmp, b, dt / 2, DIR_Z);
        semi_discrete_step(where, ctx, state, state_tmp, state, b, dt / 1, DIR_Z);
        // x-direction first
        semi_discrete_step(where, ctx, state, state, state_tmp, b, dt / 3, DIR_X);
        semi_discrete_step(where, ctx, state, state_tmp, state_tmp, b, dt / 2, DIR_X);
        semi_discrete_step(where, ctx, state, state_tmp, state, b, dt / 1, DIR_X);
    }
    if (direction_switch) {
        direction_switch = 0;
    } else {
        direction_switch = 1;
    }
}

// Perform a single semi-discretized step in time with the form:
// state_out = state_init + dt * rhs(state_forcing)
// Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step(exec_place& where, context& ctx, state_t& state_init, state_t& state_forcing,
        state_t& state_out, boundaries_t& b, double dt, int dir) {
    // Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
    tend_t tend(ctx, nx, nz, NUM_VARS);

    if (dir == DIR_X) {
        // Set the halo values  in the x-direction
        set_halo_values_x(where, ctx, state_forcing, b);
        // Compute the time tendencies for the fluid state in the x-direction
        compute_tendencies_x(where, ctx, state_forcing, tend, b);
    } else if (dir == DIR_Z) {
        // Set the halo values  in the z-direction
        set_halo_values_z(where, ctx, state_forcing);
        // Compute the time tendencies for the fluid state in the z-direction
        compute_tendencies_z(where, ctx, state_forcing, tend, b);
    }

    // Apply the tendencies to the fluid state
    ctx.parallel_for(policy(), where, tend.l.shape(), state_out.l.write(), state_init.l.read(), tend.l.read())
                    .set_symbol("apply tend")
                    ->*[=] __host__ __device__(size_t i, size_t k, size_t ll, slice<double, 3> dstate_out,
                               slice<const double, 3> dstate_init, slice<const double, 3> dtend) {
                            dstate_out(i + hs, k + hs, ll) = dstate_init(i + hs, k + hs, ll) + dt * dtend(i, k, ll);
                        };
}

// Compute the time tendencies of the fluid state using forcing in the x-direction

// First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
// Then, compute the tendencies using those fluxes
void compute_tendencies_x(exec_place& where, context& ctx, state_t& state, tend_t& tend, boundaries_t& b) {
    // Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
    //  double *flux = (double *)malloc((nx + 1) * (nz + 1) * NUM_VARS * sizeof(double));

    double dx_ = dx;
    auto hy_dens_cell = b.hy_dens_cell;
    auto hy_dens_theta_cell = b.hy_dens_theta_cell;

    auto lflux = ctx.logical_data<double>(nx + 1, nz, NUM_VARS);
    lflux.set_symbol("flux_x");

    // int i, k, ll, s, inds, indf1, indf2, indt;
    // Compute the hyperviscosity coeficient
    double hv_coef = -hv_beta * dx / (16 * dt);
    // Compute fluxes in the x-direction for each cell
    ctx.parallel_for(policy(), where, box(nx + 1, nz), state.l.read(), lflux.write())
                    .set_symbol("comp_tend_x")
                    ->*
            [=] __host__ __device__(size_t i, size_t k, slice<const double, 3> dstate, slice<double, 3> dflux)
                    {
                double d3_vals[NUM_VARS], vals[NUM_VARS];
                // Use fourth-order interpolation from four cell averages to compute the value at the interface in
                // question
                for (size_t ll = 0; ll < NUM_VARS; ll++) {
                    double stencil[4];
                    for (size_t s = 0; s < sten_size; s++) {
                        stencil[s] = dstate(i + s, k + hs, ll);
                    }
                    // Fourth-order-accurate interpolation of the state
                    vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
                    // First-order-accurate interpolation of the third spatial derivative of the state (for artificial
                    // viscosity)
                    d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
                }

                // Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
                double r = vals[ID_DENS] + hy_dens_cell(k + hs);
                double u = vals[ID_UMOM] / r;
                double w = vals[ID_WMOM] / r;
                double t = (vals[ID_RHOT] + hy_dens_theta_cell(k + hs)) / r;
                double p = C0 * pow((r * t), gamm);
                //      fprintf(stderr, "FLUX P %e\n", p);

                // Compute the flux vector
                dflux(i, k, ID_DENS) = r * u - hv_coef * d3_vals[ID_DENS];
                dflux(i, k, ID_UMOM) = r * u * u + p - hv_coef * d3_vals[ID_UMOM];
                dflux(i, k, ID_WMOM) = r * u * w - hv_coef * d3_vals[ID_WMOM];
                dflux(i, k, ID_RHOT) = r * u * t - hv_coef * d3_vals[ID_RHOT];
            };

    // Use the fluxes to compute tendencies for each cell
    ctx.parallel_for(policy(), where, tend.l.shape(), tend.l.write(), lflux.read()).set_symbol("update_tend_x")
                    ->*
            [=] __host__ __device__(size_t i, size_t k, size_t ll, slice<double, 3> dtend, slice<const double, 3> dflux) {
                dtend(i, k, ll) = -(dflux(i + 1, k, ll) - dflux(i, k, ll)) / dx_;
            };
}

// Compute the time tendencies of the fluid state using forcing in the z-direction

// First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
// Then, compute the tendencies using those fluxes
void compute_tendencies_z(exec_place& where, context& ctx, state_t& state, tend_t& tend, boundaries_t& b) {
    double dz_ = dz;

    // Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
    auto lflux = ctx.logical_data<double>(nx, nz + 1, NUM_VARS);
    lflux.set_symbol("flux_z");

    // Compute the hyperviscosity coeficient
    double hv_coef = -hv_beta * dx / (16 * dt);
    // Compute fluxes in the x-direction for each cell

    auto hy_dens_int = b.hy_dens_int;
    auto hy_dens_theta_int = b.hy_dens_theta_int;
    auto hy_pressure_int = b.hy_pressure_int;

    ctx.parallel_for(policy(), where, box(nx, nz + 1), state.l.read(), lflux.write())
                    .set_symbol("comp_tend_z")
                    ->*
            [=] __host__ __device__(size_t i, size_t k, slice<const double, 3> dstate, slice<double, 3> dflux) {
                double d3_vals[NUM_VARS], vals[NUM_VARS];
                // Use fourth-order interpolation from four cell averages to compute the value at the interface in
                // question
                for (size_t ll = 0; ll < NUM_VARS; ll++) {
                    double stencil[4];
                    for (size_t s = 0; s < sten_size; s++) {
                        stencil[s] = dstate(i + hs, k + s, ll);
                    }
                    // Fourth-order-accurate interpolation of the state
                    vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
                    // First-order-accurate interpolation of the third spatial derivative of the state
                    d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
                }

                // Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
                double r = vals[ID_DENS] + hy_dens_int(k);
                double u = vals[ID_UMOM] / r;
                double w = vals[ID_WMOM] / r;
                double t = (vals[ID_RHOT] + hy_dens_theta_int(k)) / r;
                double p = C0 * pow((r * t), gamm) - hy_pressure_int(k);

                // Compute the flux vector with hyperviscosity
                dflux(i, k, ID_DENS) = r * w - hv_coef * d3_vals[ID_DENS];
                dflux(i, k, ID_UMOM) = r * w * u - hv_coef * d3_vals[ID_UMOM];
                dflux(i, k, ID_WMOM) = r * w * w + p - hv_coef * d3_vals[ID_WMOM];
                dflux(i, k, ID_RHOT) = r * w * t - hv_coef * d3_vals[ID_RHOT];
            };

    // Use the fluxes to compute tendencies for each cell
    ctx.parallel_for(policy(), where, tend.l.shape(), tend.l.write(), lflux.read(), state.l.read())
                    .set_symbol("update_tend_z")
                    ->*[=] __host__ __device__(size_t i, size_t k, size_t ll, slice<double, 3> dtend,
                               slice<const double, 3> dflux, slice<const double, 3> dstate) {
                            dtend(i, k, ll) = -(dflux(i, k + 1, ll) - dflux(i, k, ll)) / dz_;

                            if (ll == ID_WMOM) {
                                dtend(i, k, ll) -= dstate(i + hs, k + hs, ID_DENS);
                            }
                        };
}

void set_halo_values_x(exec_place& where, context& ctx, state_t& state, boundaries_t& b) {
    //    int k, ll, ind_r, ind_u, ind_t, i;

    double dz_ = dz;
    int nx_ = nx;
    int k_beg_ = k_beg;

    auto hy_dens_theta_cell = b.hy_dens_theta_cell;
    auto hy_dens_cell = b.hy_dens_cell;

    ctx.parallel_for(policy(), where, box(nz, NUM_VARS), state.l.rw()).set_symbol("set halo x")
                    ->*[=] __host__ __device__(size_t k, size_t ll, slice<double, 3> dstate) {
                            dstate(0, k + hs, ll) = dstate(nx_ + hs - 2, k + hs, ll);
                            dstate(1, k + hs, ll) = dstate(nx_ + hs - 1, k + hs, ll);
                            dstate(nx_ + hs, k + hs, ll) = dstate(hs, k + hs, ll);
                            dstate(nx_ + hs + 1, k + hs, ll) = dstate(hs + 1, k + hs, ll);
                        };

    if (myrank == 0) {
        ctx.parallel_for(
                   policy(), where, box(nz, hs), state.l.rw())
                        .set_symbol("set halo x(2)")
                        ->*
                [=] __host__ __device__(size_t k, size_t i, slice<double, 3> dstate) {
                    double z = ((double) k_beg_ + (double) k + 0.5) * dz_;
                    if (fabs(z - 3.0 * zlen / 4.0) <= zlen / 16.0) {
                        dstate(i, k + hs, ID_UMOM) = (dstate(i, k + hs, ID_DENS) + hy_dens_cell(k + hs)) * 50.;
                        dstate(i, k + hs, ID_RHOT) =
                                (dstate(i, k + hs, ID_DENS) + hy_dens_cell(k + hs)) * 298. - hy_dens_theta_cell(k + hs);
                    }
                };
    }
}

// Set this task's halo values in the z-direction.
// decomposition in the vertical direction.
void set_halo_values_z(exec_place& where, context& ctx, state_t& state) {
    //  int i, ll;
    //  const double mnt_width = xlen / 8;
    //  double x, xloc, mnt_deriv;

    int nz_ = nz;

    ctx.parallel_for(policy(), where, box(nx + 2 * hs, NUM_VARS), state.l.rw()).set_symbol("set halo z")
                    ->*[=] __host__ __device__(size_t i, size_t ll, slice<double, 3> dstate) {
                            if (ll == ID_WMOM) {
                                dstate(i, 0, ll) = 0.;
                                dstate(i, 1, ll) = 0.;
                                dstate(i, nz_ + hs, ll) = 0.;
                                dstate(i, nz_ + hs + 1, ll) = 0.;
                            } else {
                                dstate(i, 0, ll) = dstate(i, hs, ll);
                                dstate(i, 1, ll) = dstate(i, hs, ll);
                                dstate(i, nz_ + hs, ll) = dstate(i, nz_ + hs - 1, ll);
                                dstate(i, nz_ + hs + 1, ll) = dstate(i, nz_ + hs - 1, ll);
                            }
                        };
}

void init(exec_place& where, context& ctx, state_t& state, state_t& state_tmp, boundaries_t& b) {
    int i_end;
    double nper;

    // Set the cell grid size
    dx = xlen / nx_glob;
    dz = zlen / nz_glob;

    nranks = 1;
    myrank = 0;

    // For simpler version, replace i_beg = 0, nx = nx_glob, left_rank = 0, right_rank = 0;

    nper = ((double) nx_glob) / nranks;
    i_beg = round(nper * (myrank));
    i_end = round(nper * ((myrank) + 1)) - 1;
    nx = i_end - i_beg + 1;
    left_rank = myrank - 1;
    if (left_rank == -1)
        left_rank = nranks - 1;
    right_rank = myrank + 1;
    if (right_rank == nranks)
        right_rank = 0;

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    k_beg = 0;
    nz = nz_glob;

    // Define the maximum stable time step based on an assumed maximum wind speed
    dt = std::min(dx, dz) / max_speed * cfl;
    // Set initial elapsed model time and output_counter to zero
    etime = 0.;
    output_counter = 0.;

    // Display grid information

    printf("nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf("dx,dz: %lf %lf\n", dx, dz);
    printf("dt: %lf\n", dt);

    int nqpoints = 3;
    double qpoints[3] = { 0.112701665379258311482073460022E0, 0.500000000000000000000000000000E0,
        0.887298334620741688517926539980E0 };
    double qweights[3] = { 0.277777777777777777777777777779E0, 0.444444444444444444444444444444E0,
        0.277777777777777777777777777779E0 };

    auto lqweights = ctx.logical_data(qweights);
    auto lqpoints = ctx.logical_data(qpoints);

    auto dx_ = dx;
    auto dz_ = dz;
    auto i_beg_ = i_beg;
    auto k_beg_ = k_beg;

    //////////////////////////////////////////////////////////////////////////
    // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
    //////////////////////////////////////////////////////////////////////////
    ctx.parallel_for(policy(), where, box(nx + 2 * hs, nz + 2 * hs), state.l.write(), lqweights.read(), lqpoints.read())
                    .set_symbol("init_fluid_cells")
                    ->*
            [=] __host__ __device__(
                    size_t i, size_t k, slice<double, 3> hstate, slice<const double> qweights, slice<const double> qpoints) {
                // Initialize the state to zero
                for (size_t ll = 0; ll < NUM_VARS; ll++) {
                    hstate(i, k, ll) = 0.;
                }
                // Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
                for (size_t kk = 0; kk < nqpoints; kk++) {
                    for (size_t ii = 0; ii < nqpoints; ii++) {
                        // Compute the x,z location within the global domain based on cell and quadrature index
                        double x = ((double) i_beg_ + (double) i - (double) hs + 0.5) * dx_ + (qpoints[ii] - 0.5) * dx_;
                        double z = ((double) k_beg_ + (double) k - (double) hs + 0.5) * dz_ + (qpoints[kk] - 0.5) * dz_;

                        // Set the fluid state based on the user's specification (default is injection in this example)
                        double r, u, w, t, hr, ht;
                        injection(x, z, r, u, w, t, hr, ht);

                        // Store into the fluid state array
                        hstate(i, k, ID_DENS) += r * qweights[ii] * qweights[kk];
                        hstate(i, k, ID_UMOM) += (r + hr) * u * qweights[ii] * qweights[kk];
                        hstate(i, k, ID_WMOM) += (r + hr) * w * qweights[ii] * qweights[kk];
                        hstate(i, k, ID_RHOT) += ((r + hr) * (t + ht) - hr * ht) * qweights[ii] * qweights[kk];
                    }
                }
            };

    ctx.parallel_for(policy(), where, state.l.shape(), state.l.read(), state_tmp.l.write())
                    .set_symbol("init_fluid_cells_cpy")
                    ->*[] __host__ __device__(size_t i, size_t k, size_t ll, slice<const double, 3> hstate,
                               slice<double, 3> hstate_tmp) { hstate_tmp(i, k, ll) = hstate(i, k, ll); };

    // Compute the hydrostatic background state over vertical cell averages
    ctx.parallel_for(policy(), where, b.lhy_dens_cell.shape(), b.lhy_dens_cell.write(), b.lhy_dens_theta_cell.write())
                    .set_symbol("init_hydro_background")
                    ->*[=] __host__ __device__(size_t k, slice<double> hy_dens_cell, slice<double> hy_dens_theta_cell) {
                            hy_dens_cell(k) = 0.;
                            hy_dens_theta_cell(k) = 0.;
                            for (int kk = 0; kk < nqpoints; kk++) {
                                double z = (k_beg_ + (double) k - (double) hs + 0.5) * dz_;

                                // Set the fluid state based on the user's specification (default is injection in this
                                // example)
                                double r, u, w, t, hr, ht;
                                injection(0., z, r, u, w, t, hr, ht);

                                hy_dens_cell(k) += hr * qweights[kk];
                                hy_dens_theta_cell(k) += hr * ht * qweights[kk];
                            }
                        };

    // Compute the hydrostatic background state at vertical cell interfaces
    ctx.parallel_for(policy(), where, b.lhy_dens_int.shape(), b.lhy_dens_int.write(), b.lhy_dens_theta_int.write(),
               b.lhy_pressure_int.write())
                    .set_symbol("init_hydro_background_interfaces")
                    ->*[=] __host__ __device__(size_t k, slice<double> hy_dens_int, slice<double> hy_dens_theta_int,
                               slice<double> hy_pressure_int) {
                            double z = ((double) k_beg_ + (double) k) * dz_;

                            // Set the fluid state based on the user's specification (default is injection in this
                            // example)
                            double r, u, w, t, hr, ht;
                            injection(0., z, r, u, w, t, hr, ht);

                            hy_dens_int(k) = hr;
                            hy_dens_theta_int(k) = hr * ht;
                            hy_pressure_int(k) = C0 * pow((hr * ht), gamm);
                        };

    b.flhy_dens_cell = ctx.freeze(b.lhy_dens_cell);
    b.flhy_dens_theta_cell =  ctx.freeze(b.lhy_dens_theta_cell);
    b.flhy_dens_int = ctx.freeze(b.lhy_dens_int);
    b.flhy_dens_theta_int = ctx.freeze(b.lhy_dens_theta_int);
    b.flhy_pressure_int = ctx.freeze(b.lhy_pressure_int);

    b.flhy_dens_cell.set_automatic_unfreeze();
    b.flhy_dens_theta_cell.set_automatic_unfreeze();
    b.flhy_dens_int.set_automatic_unfreeze();
    b.flhy_dens_theta_int.set_automatic_unfreeze();
    b.flhy_pressure_int.set_automatic_unfreeze();

    auto dplace = where.is_grid()?
                      data_place::composite(policy(), where.as_grid())
                      :where.affine_data_place();

    b.hy_dens_cell = b.flhy_dens_cell.get(dplace).first;
    b.hy_dens_theta_cell = b.flhy_dens_theta_cell.get(dplace).first;
    b.hy_dens_int = b.flhy_dens_int.get(dplace).first;
    b.hy_dens_theta_int = b.flhy_dens_theta_int.get(dplace).first;
    b.hy_pressure_int = b.flhy_pressure_int.get(dplace).first;

    // Ensure get operations are completed
    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));
}

// This test case is initially balanced but injects fast, cold air from the left boundary near the model top
// x and z are input coordinates at which to sample
// r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
// hr and ht are output background hydrostatic density and potential temperature at that location
__host__ __device__ void injection(
        double /* unused x */, double z, double& r, double& u, double& w, double& t, double& hr, double& ht) {
    hydro_const_theta(z, hr, ht);
    r = 0.;
    t = 0.;
    u = 0.;
    w = 0.;
}

// Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
// z is the input coordinate
// r and t are the output background hydrostatic density and potential temperature
__host__ __device__ void hydro_const_theta(double z, double& r, double& t) {
    const double theta0 = 300.;  // Background potential temperature
    const double exner0 = 1.;    // Surface-level Exner pressure
    double p, exner, rt;
    // Establish hydrostatic balance first using Exner pressure
    t = theta0;                                 // Potential Temperature at z
    exner = exner0 - grav * z / (cp * theta0);  // Exner pressure at z
    p = p0 * pow(exner, (cp / rd));             // Pressure at z
    rt = pow((p / C0), (1. / gamm));            // rho*theta at z
    r = rt / t;                                 // Density at z
}

#ifdef HAVE_NETCDF
// Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
// The file I/O uses netcdf, the only external library required for this mini-app.
// If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
void output(context& ctx, state_t& state, boundaries_t& b, double etime) {
    auto hy_dens_cell = b.hy_dens_cell;
    auto hy_dens_theta_cell = b.hy_dens_theta_cell;

    ctx.host_launch(state.l.read())
                    ->*[=](slice<const double, 3> hstate) {
                            int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid,
                                    t_varid, dimids[3];
                            int i, k;

                            size_t st1[1], ct1[1], st3[3], ct3[3];

                            // Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
                            double *dens, *uwnd, *wwnd, *theta;
                            double* etimearr;
                            // Inform the user

                            printf("*** OUTPUT ***\n");

                            // Allocate some (big) temp arrays
                            dens = (double*) malloc(nx * nz * sizeof(double));
                            uwnd = (double*) malloc(nx * nz * sizeof(double));
                            wwnd = (double*) malloc(nx * nz * sizeof(double));
                            theta = (double*) malloc(nx * nz * sizeof(double));
                            etimearr = (double*) malloc(1 * sizeof(double));

                            // If the elapsed time is zero, create the file. Otherwise, open the file
                            if (etime == 0) {
                                // Create the file
                                ncwrap(nc_create("reference.nc", NC_CLOBBER, &ncid), __LINE__);

                                // Create the dimensions
                                ncwrap(nc_def_dim(ncid, "t", NC_UNLIMITED, &t_dimid), __LINE__);
                                ncwrap(nc_def_dim(ncid, "x", nx_glob, &x_dimid), __LINE__);
                                ncwrap(nc_def_dim(ncid, "z", nz_glob, &z_dimid), __LINE__);

                                // Create the variables
                                dimids[0] = t_dimid;
                                ncwrap(nc_def_var(ncid, "t", NC_DOUBLE, 1, dimids, &t_varid), __LINE__);

                                dimids[0] = t_dimid;
                                dimids[1] = z_dimid;
                                dimids[2] = x_dimid;

                                ncwrap(nc_def_var(ncid, "dens", NC_DOUBLE, 3, dimids, &dens_varid), __LINE__);
                                ncwrap(nc_def_var(ncid, "uwnd", NC_DOUBLE, 3, dimids, &uwnd_varid), __LINE__);
                                ncwrap(nc_def_var(ncid, "wwnd", NC_DOUBLE, 3, dimids, &wwnd_varid), __LINE__);
                                ncwrap(nc_def_var(ncid, "theta", NC_DOUBLE, 3, dimids, &theta_varid), __LINE__);

                                // End "define" mode
                                ncwrap(nc_enddef(ncid), __LINE__);
                            } else {
                                // Open the file
                                ncwrap(nc_open("reference.nc", NC_WRITE, &ncid), __LINE__);

                                // Get the variable IDs
                                ncwrap(nc_inq_varid(ncid, "dens", &dens_varid), __LINE__);
                                ncwrap(nc_inq_varid(ncid, "uwnd", &uwnd_varid), __LINE__);
                                ncwrap(nc_inq_varid(ncid, "wwnd", &wwnd_varid), __LINE__);
                                ncwrap(nc_inq_varid(ncid, "theta", &theta_varid), __LINE__);
                                ncwrap(nc_inq_varid(ncid, "t", &t_varid), __LINE__);
                            }

                            // Store perturbed values in the temp arrays for output
                            for (k = 0; k < nz; k++) {
                                for (i = 0; i < nx; i++) {
                                    auto r = hstate(i + hs, k + hs, ID_DENS);
                                    auto u = hstate(i + hs, k + hs, ID_UMOM);
                                    auto w = hstate(i + hs, k + hs, ID_WMOM);
                                    auto t = hstate(i + hs, k + hs, ID_RHOT);

                                    dens[k * nx + i] = r;
                                    uwnd[k * nx + i] = u / (hy_dens_cell(k + hs) + r);
                                    wwnd[k * nx + i] = w / (hy_dens_cell(k + hs) + r);
                                    theta[k * nx + i] = (t + hy_dens_theta_cell[k + hs]) / (hy_dens_cell(k + hs) + r) -
                                                        hy_dens_theta_cell(k + hs) / hy_dens_cell(k + hs);
                                    //          fprintf(stderr, "DUMP DENS(%d, %d) = %e\n", i, k, r);
                                }
                            }

                            // Write the grid data to file with all the processes writing collectively
                            st3[0] = num_out;
                            st3[1] = k_beg;
                            st3[2] = i_beg;
                            ct3[0] = 1;
                            ct3[1] = nz;
                            ct3[2] = nx;

                            ncwrap(nc_put_vara_double(ncid, dens_varid, st3, ct3, dens), __LINE__);
                            ncwrap(nc_put_vara_double(ncid, uwnd_varid, st3, ct3, uwnd), __LINE__);
                            ncwrap(nc_put_vara_double(ncid, wwnd_varid, st3, ct3, wwnd), __LINE__);
                            ncwrap(nc_put_vara_double(ncid, theta_varid, st3, ct3, theta), __LINE__);

                            // Only the master process needs to write the elapsed time
                            // write elapsed time to file

                            st1[0] = num_out;
                            ct1[0] = 1;
                            etimearr[0] = etime;
                            ncwrap(nc_put_vara_double(ncid, t_varid, st1, ct1, etimearr), __LINE__);

                            // Close the file
                            ncwrap(nc_close(ncid), __LINE__);

                            // Increment the number of outputs
                            num_out = num_out + 1;

                            // Deallocate the temp arrays
                            free(dens);
                            free(uwnd);
                            free(wwnd);
                            free(theta);
                            free(etimearr);
                        };
}

// Error reporting routine for the NetCDF I/O
void ncwrap(int ierr, int line) {
    if (ierr != NC_NOERR) {
        printf("NetCDF Error at line: %d\n", line);
        printf("%s\n", nc_strerror(ierr));
        exit(-1);
    }
}
#endif  // HAVE_NETCDF
