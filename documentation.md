# The MiniWeather Mini App

Author: Matt Norman, Oak Ridge National Laboratory, https://mrnorman.github.io

# Introduction

The miniWeather code mimics the basic dynamics seen in atmospheric weather and climate. The dynamics themselves are dry compressible, stratified, non-hydrostatic flows dominated by buoyant forces that are relatively small perturbations on a hydrostatic background state. The equations in this code themselves form the backbone of pretty much all fluid dynamics codes, and this particular flavor forms the base of all weather and climate modeling.

With about 500 total lines of code (and only about 200 lines that you care about), it serves as an approachable place to learn parallelization and porting using MPI + X, where X is OpenMP, OpenACC, CUDA, or potentially other approaches to CPU and accelerated parallelization. The code uses periodic boundary conditions in the x-direction and solid wall boundary conditions in the z-direction. 

## Brief Description of the Code

### Domain Parameters

A fuller description of the science, math, and dynamics are play are given later, but this section is reserved to describing some of the main variables and flow in the code. The code is decomposed in two spatial dimensions, x and z, with `nx_glob` and `nz_glob` cells in the global domain and nx and nz cells in the local domain, using straightforward domain decomposition for MPI-level parallelization. The global domain is of size xlen and zlen meters, and hs “halo” cells are appended to both sides of each dimension for convenience in forming stencils of cells for reconstruction.

### Fluid State Variables

There are four main arrays used in this code: `state`, `state_tmp`, `flux`, and `tend`, and the dimensions for each are given in the code upon declaration in the comments. Each of these arrays is described briefly below:

* `state`: This is the fluid state at the current time step, and it is the only array that persists from one time step to the next. The other four are only used within the calculations to advance the model to the next time step. The fluid state describes the average state over each cell area in the spatial domain. This variable contains four fluid states, which are the traditional mass, momenta, and thermodynamic quantities of most fluid models:
  1. Density (`ID_DENS`): The 2-D density of the fluid, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\rho" title="\large \rho" />, in <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\text{kg}\&space;\text{m}^{-2}" title="\large \text{kg}\ \text{m}^{-2}" /> (note this is normally <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\text{kg}\&space;\text{m}^{-3}" title="\large \text{kg}\ \text{m}^{-3}" />, but this is a 2-D model, not 3-D)
  2. U-momentum (`ID_UMOM`): The momentum per unit area of the fluid in the x-direction calculated as <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\rho&space;u" title="\large \rho u" />, where u is the x-direction wind velocity. The units are <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\text{kg}\&space;\text{m}^{-1}\&space;\text{s}^{-1}" title="\large \text{kg}\ \text{m}^{-1}\ \text{s}^{-1}" />. Note that to get true momentum, you must integrate over the cell.
  2. W-momentum (`ID_WMOM`): The momentum per unit area of the fluid in the z-direction calculated as <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\rho&space;w" title="\large \rho w" />, where w is the z-direction wind velocity. The units are <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\text{kg}\&space;\text{m}^{-1}\&space;\text{s}^{-1}" title="\large \text{kg}\ \text{m}^{-1}\ \text{s}^{-1}" />. Note that to get true momentum, you must integrate over the cell.
  4. Potential Temperature (`ID_RHOT`): The product of density and potential temperature, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\rho&space;\theta" title="\large \rho \theta" />, where <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\theta=T\left(P_{0}/P\right)^{R_{d}/c_{p}}" title="\large \theta=T\left(P_{0}/P\right)^{R_{d}/c_{p}}" />, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;P_{0}=10^{5}\,\text{Pa}" title="\large P_{0}=10^{5}\,\text{Pa}" />, T is the true temperature, and <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;R_d" title="\large R_d" /> and<img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;c_p" title="\large c_p" /> are the dry air constant and specific heat at constant pressure for dry air, respectively. The units of this quantity are <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\text{K}\,\text{kg}\,\text{m}^{-2}" title="\large \text{K}\,\text{kg}\,\text{m}^{-2}" />.
* `state_tmp`: This is a temporary copy of the fluid state used in the Runge-Kutta integration to keep from overwriting the state at the beginning of the time step, and it has the same units and meaning.
* `flux`: This is fluid state at cell boundaries in the x- and z-directions, and the units and meanings are the same as for `state` and `state_tmp`. In the x-direction update, the values of `flux` at indices `i` and `i+1` represents the fluid state at the left- and right-hand boundaries of cell `i`. The indexing is analagous in the z-direction. The fluxes are used to exchange fluid properties with neighboring cells.
* `tend`: This is the time tendency of the fluid state <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\partial\mathbf{q}/\partial&space;t" title="\large \partial\mathbf{q}/\partial t" />, where <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\mathbf{q}" title="\large \mathbf{q}" /> is the the state vector, and as the name suggests, it has the same meaning and units as state, except per unit time (appending <img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\large&space;\text{s}^{-1}" title="\large \text{s}^{-1}" /> to the units). In the Finite-Volume method, the time tendency of a cell is equivalent to the divergence of the flux across a cell.

### Overal Model Flow

This code follows a traditional Finite-Volume control flow.

To compute the time tendency, given an initial state at the beginning of a time step that contains cell-averages of the fluid state, the value of the state at cell boundaries is reconstructed using a stencil. Then, a viscous term is added to the fluid state at the cell boundaries to improve stability. Next, the tendencies are computed as the divergence of the flux across the cell. Finally, the tendencies are applied to the fluid state.

Once the time tendency is computed, the fluid PDEs are essentially now cast as a set of ODEs varying only in time, and this is called the “semi-discretized” form of the equations. To solve the equations in time, an ODE solver is used, in this case, a Runge-Kutta method. Finally, at the highest level, the equations are split into separate solves in each dimension, x and z.

# Compiling and Running the Code

## Required Libraries

To run everything in this code, you need MPI, parallel-netcdf, ncview, an OpenACC compiler (PGI,GNU,Cray) for OpenACC work, an OpenMP offload compiler (IBM XL, Cray) for OpenMP offload work. The serial code still uses MPI because I figured it was best to start with parallel I/O in place so that it's ready to go when the MPI implementation is done.

You can download parallel-netcdf from: https://trac.mcs.anl.gov/projects/parallel-netcdfAlso, you can download a free version of PGI Community Edition (which has OpenACC implemented) from: https://www.pgroup.com/products/community.htm. You'll probably also want to have `ncview` installed so you can easily view the NetCDF files. Ncview also makes it easy to see a movie of the solution as it evolves in time: http://meteora.ucsd.edu/~pierce/ncview_home_page.html

## Directories and Compiling

There are four main directories in the mini app: (1) a Fortran source directory; (2) a C source directory; (3) a C++ source directory; and (4) a documentation directory. The C code is technically C++ code but only because I wanted to use the ampersand pass by reference notation rather than the hideious C asterisks.

To compile the code, first edit the `Makefile` and change the flags to point to your parallel-netcdf installation as well as change the flags based on which compiler you are using. There are five versions of the code in C and Fortran: serial, mpi, mpi+openmp, and mpi+openacc, mpi+openmp4.5. The filenames make it clear which file is associated with which programming paradigm. To make all of these at once, simply type `make`. To make them individually, you can type `make [serial|mpi|openmp|openacc|openmp45]`.

## Altering the Code's Configurations

There are four aspects of the configuration you can edit easily, and they are clearly labeled in the code as “USER-CONFIGURABLE PARAMETERS”. These include: (1) the number of cells to use (`nx_glob` and `nz_glob`); (2) the amount of time to simulate (`sim_time`); (3) the frequency to output data to file ('output_freq'); and (4) the initial data to use (`data_spec_int`). 

## Running the Code

To run the code, simply call `mpirun -n [# ranks] ./mini_weather_[version]`. Since parameters are set in the code itself, you don't need to pass any parameters. Some machines use different tools instead of mpirun (e.g., OLCF's Titan uses `aprun`).

## Viewing the Output

The file I/O is done in the netCDF format: (https://www.unidata.ucar.edu/software/netcdf). To me, the easiest way to view the data is to use a tool called “ncview” (http://meteora.ucsd.edu/~pierce/ncview_home_page.html). To use it, you can simply type `ncview output.nc`, making sure you have X-forwarding enabled in your ssh session. Further, you can call `ncview -frames output.nc`, and it will dump out all of your frames in the native resolution you're viewing the data in, and you you can render a movie with tools like `ffmpeg`. 

# Parallelization



