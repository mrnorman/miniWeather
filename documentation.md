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

### C and Fortran

To compile the code, first edit the `Makefile` and change the flags to point to your parallel-netcdf installation as well as change the flags based on which compiler you are using. There are five versions of the code in C and Fortran: serial, mpi, mpi+openmp, and mpi+openacc, mpi+openmp4.5. The filenames make it clear which file is associated with which programming paradigm. To make all of these at once, simply type `make`. To make them individually, you can type:

```
make [serial|mpi|openmp|openacc|openmp45]
```

### C++

For the C++ code, there are three configurations: serial, mpi, and mpi+`parallel_for`. The latter uses the typical C++ kernel-launching approach, which is essentially CUDA with greater portability for multiple backends.

```
make [serial|mpi|parallefor]
```

## Altering the Code's Configurations

There are four aspects of the configuration you can edit easily, and they are clearly labeled in the code as “USER-CONFIGURABLE PARAMETERS”. These include: (1) the number of cells to use (`nx_glob` and `nz_glob`); (2) the amount of time to simulate (`sim_time`); (3) the frequency to output data to file ('output_freq'); and (4) the initial data to use (`data_spec_int`). 

## Running the Code

To run the code, simply call:

```
mpirun -n [# ranks] ./mini_weather_[version]
```

Since parameters are set in the code itself, you don't need to pass any parameters. Some machines use different tools instead of mpirun (e.g., OLCF's Titan uses `aprun`).

## Viewing the Output

The file I/O is done in the netCDF format: (https://www.unidata.ucar.edu/software/netcdf). To me, the easiest way to view the data is to use a tool called “ncview” (http://meteora.ucsd.edu/~pierce/ncview_home_page.html). To use it, you can simply type `ncview output.nc`, making sure you have X-forwarding enabled in your ssh session. Further, you can call `ncview -frames output.nc`, and it will dump out all of your frames in the native resolution you're viewing the data in, and you you can render a movie with tools like `ffmpeg`. 

# Parallelization

This code was designed to parallelize with MPI first and then OpenMP, OpenACC, OpenMP offlaod, or `parallel_for` next, but you can always parallelize with OpenMP or OpenACC without MPI if you want. But it is rewarding to be able to run it on multiple nodes at higher resolution for more and sharper eddies in the dynamics.

As you port the code, you'll want to change relatively little code at a time, re-compile, re-run, and look at the output to see that you're still getting the right answer. There are advantages to using a visual tool to check the answer (e.g., `ncview`), as it can sometimes give you clues as to why you're not getting the right answer. 

Note that you only need to make changes code within the first 450 source lines for C and Fortran, and each loop that needs threading is decorated with a `// THREAD ME` comment. Everything below that is initialization and I/O code that doesn't need to be parallelized (unless you want to) for C and Fortran directives-based approaches.

For the C++ code, you will need to work with the initialization and File I/O code, but everything you need to do is explicitly guided via comments in the code.

## Indexing

The code makes room for so-called “halo” cells in the fluid state. This is a common practice in any algorithm that uses stencil-based reconstruction to estimate variation within a domain. In this code, there are `hs` halo cells on either side of each spatial dimension, and I pretty much hard-code `hs=2`.

### Fortran

In the Fortran code's fluid state (`state`), the x- and z-dimensions are dimensioned as multi-dimensional arrays that range from `1-hs:nx+hs`. In the x-direction, `1-hs:0` belong to the MPI task to the left, cells `1:nx` belong to the current MPI task, and `nx+1:nx+hs` belong to the MPI task to the right. In the z-dimension, `1-hs:0` are artificially set to mimic a solid wall boundary condition at the bottom, and `nz+1:nz+hs` are the same for the top boundary. The cell-interface fluxes (`flux`) are dimensioned as `1:nx+1` and `1:nz+1` in the x- and z-directions, and the cell average tendencies (`tend`) are dimensioned `1:nx` and `1:nz` in the x- and z-directions. The cell of index `i` will have left- and right-hand interface fluxes of index `i` and `i+1`, respectively, and it will be evolved by the tendency at index `i`. The analog of this is also true in the z-direction.

### C

In the C code, the fluid `state` array is dimensioned to size `nz+2*hs` and `nx+2*hs` in the x- and z-directions. In the x-direction, cells `0` to `hs-1` belong to the left MPI task, cells `hs` to `nx+hs-1` belong to the current MPI taks, and cells `nx+hs` to `nx+2*hs-1` belong to the right MPI task. The z-direction's halo cells are used to mimic solid wall boundaries. The cell-interface fluxes (`flux`) are dimensioned as `nx+1` and `nz+1` in the x- and z-directions, and the cell average tendencies (`tend`) are dimensioned `nx` and `nz` in the x- and z-directions. The cell of index `i+hs` will have left- and right-hand interface fluxes of index `i` and `i+1`, respectively, and it will be evolved by the tendency at index `i`. The analog of this is also true in the z-direction.

### C++

The C++ indexing is the same as the C indexing, but instead of having to flatten array indices into a single dimension like the C code, multi-dimensional arrays are used with `()` indexing syntax and the right-most index varying the fastest.

## MPI Domain Decomposition

This code was designed to use domain decomposition, where each MPI rank “owns” a certain set of cells in the x-direction and contains two “halo” cells from the left- and right-hand MPI tasks in the x-direction as well. The domain is only decomposed in the x-direction and not the z-direction for simplicity.

**IMPORTANT**: Please be sure to set `nranks`, `myrank`, `nx`, `i_beg`, `left_rank`, and `right_rank`. These are clearly marked in the serial source code. You can set more variables, but these are used elsewhere in the code (particularly in the parallel file I/O), so they must be set.

To parallelize with MPI, there are only two places in the code that need to be altered. The first is the initialization, a subroutine / function named `init`, where you must determine the number of ranks, you process's rank, the beginning index of your rank's first cell in the x-direction, the number of x-direction cells your rank will own, and the MPI rank IDs that are to your left and your right. Because the code is periodic in the x-direction, your left and right neighboring ranks will wrap around. For instance, if your are rank `0`, your left-most rank will be `nranks-1`.

The second place is in the routine that sets the halo values in the x-direction. In this routine, you need to:

1. Create MPI data buffers (at the same place the other arrays are declared) to hold the data that needs to be sent and received, allocate them in the `init()` routine, and deallocate them in the `finalize()` routine.

2. Pack the data you need to send to your left and right MPI neighbors

3. Send the data to your left and right MPI neighbors

4. Receive the data from your left and right MPI neighbors

5. Unpack the data from your left and right neighbors and place the data into your MPI rank's halo cells. 

Once you complete this, the code will be fully parallelized in MPI. Both of the places you need to add code for MPI are marked in the serial code, and there are some extra hints in the `set_halo_values_x()` routine as well.

## OpenMP CPU Threading

For the OpenMP code, you basically need to decorate the loops with `omp parallel do` in Fortran or `omp parallel for` in C, and pay attention to any variables you need to make `private()` so that each thread has its own copy. Keep in mind that OpenMP works best on “outer” loops rather than “inner” loops. Also, for sake of performance, there are a couple of instances where it is wise to use the “collapse” clause because the outermost loop is not large enough to support the number of threads most CPUs have.

## OpenACC Accelerator Threading

The OpenACC approach will differ depending on whether you're in Fortran or C. Just a forewarning, OpenACC is much more convenient in Fortran when it comes to data movement because in Fortran, the compiler knows how big your arrays are, and therefore the compiler can (and does) create all of the data movement for you (NOTE: This is true for PGI and Cray but not for GNU at the moment). All you have to do is optimize the data movement after the fact.

### Fortran Code

The OpenACC parallelization is a bit more involved when it comes to performance. But, it's a good idea to just start with the kernels themselves, since the compiler will generate all of your data statements for you on a per-kernel basis. You need to pay attention to private variables here as well. Only arrays need to be privatized. Scalars are automatically privatized for you.

Once you're getting the right answer with the kernels on the GPU, you can look at optimizing data movement by putting in data statements. I recommend putting data statements for the `state`, `tend`, `flux`, `hy_*`, and the MPI buffers (`sendbuf_l`, `sendbuf_r`, `recvbuf_l`, and `recvbuf_r`) around the main time stepping loop. Then, you need to move the data to the host before sending MPI data, back to the device once you receive MPI data, and to the host before file I/O.

### C Code

In the C code, you'll need to put in manual `copy()`, `copyin()`, and `copyout()` statements on a **per-kernel basis**, and you'll need to explicitly declare the size of each array as well.

**IMPORTANT**: The syntax for data movement in C will seem odd to you. The syntax is:

```C
#pragma acc data copy( varname[ starting_index : size_of_transfer ] )
```

So, for instance, if you send a variable, `var`, of size `n` to the GPU, you will say, `#pragma acc data copyin(var[0:n])`. Many would expect it to look like an array slice (e.g., `(0:n-1)`), but it is not. 

Other than this, the approach is the same as with the Fortran case.

## C++ Performance Portability

The C++ code is in the `cpp` directory, and it uses a custom multi-dimensional `Array` class from `Array.h` for large global variables and Static Array (`SArray`) class in `SArray.h`. For adding MPI to the serial code, please follow the instructions in the above MPI section. The primary purpose of the C++ code is to get used to what performance portability looks like in C++, and this is moving from the `miniWeather_mpi.cpp` code to the `miniWeather_mpi_parallelfor.cpp` code, where you change all of the loops into `parallel_for` kernel launches, similar to the [Kokkos](https://github.com/kokkos/kokkos) syntax. As an example, of transforming a set of loops into `parallel_for`, consider the following code:

```C++
inline void applyTendencies(realArr &state2, real const c0, realArr const &state0,
                                             real const c1, realArr const &state1,
                                             real const ct, realArr const &tend,
                                             Domain const &dom) {
  for (int l=0; l<numState; l++) {
    for (int k=0; k<dom.nz; k++) {
      for (int j=0; j<dom.ny; j++) {
        for (int i=0; i<dom.nx; i++) {
          state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                                     c1 * state1(l,hs+k,hs+j,hs+i) +
                                     ct * dom.dt * tend(l,k,j,i);
        }
      }
    }
  }
}
```

will become:

```C++
inline void applyTendencies(realArr &state2, real const c0, realArr const &state0,
                                             real const c1, realArr const &state1,
                                             real const ct, realArr const &tend,
                                             Domain const &dom) {
  // for (int l=0; l<numState; l++) {
  //   for (int k=0; k<dom.nz; k++) {
  //     for (int j=0; j<dom.ny; j++) {
  //       for (int i=0; i<dom.nx; i++) {
  yakl::parallel_for( numState*dom.nz*dom.ny*dom.nx , YAKL_LAMBDA (int iGlob) {
    int l, k, j, i;
    unpackIndices(iGlob,numState,dom.nz,dom.ny,dom.nx,l,k,j,i);
    // LOOP BODY BEGINS HERE
    state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                               c1 * state1(l,hs+k,hs+j,hs+i) +
                               ct * dom.dt * tend(l,k,j,i);
  }); 
}
```


For a fuller description of how to move loops to parallel_for, please see the following webpage:

https://github.com/mrnorman/YAKL/wiki/CPlusPlus-Performance-Portability-For-OpenACC-and-OpenMP-Folks

I strongly recommend moving to `parallel_for` while compiling for the CPU so you don't have to worry about separate memory address spaces at the same time. Be sure to use array bounds checking during this process to ensure you don't mess up the indexing in the `parallel_for` launch. You can do this by adding `-DARRAY_DEBUG` to the `CXX_FLAGS` in your `Makefile`. After you've transformed all of the for loops to `parallel_for`, you can deal with the complications of separate memory spaces.

### GPU Mofifications

First, you'll have to pay attention to asynchronicity. `parallel_for` is asynchronous, and therefore, you'll need to add `yakl::fence()` in two places: (1) MPI ; and (2) File I/O.

Next, if a `parallel_for`'s kernel uses variables with global scope, which it will in this code, you will get a compile time error when running on the GPU. C++ Lambdas do not capture variables with global scope, and therefore, you'll be using CPU copies of that data, which nvcc does not allow. The most convenient way to handle this is to create local references as follows:

```C++
auto &varName = ::varName;
```

The `::varName` syntax it telling the compiler to look in the global namespace for `varName` rather than the local namespace, which would technically be a variable referencing itself, which is not legal C++.

This process will be tedious, but it something you nearly always have to do in C++ performance portability approaches. So it's good to get used to doing it. You will run into similar issues if you attempt to __use data from your own class__ because the `this` pointer is typically into CPU memory because class objects are often allocated on the CPU. This can also be circumvented via local references in the same manner as above.

You have to put `YAKL_INLINE` in front of the following functions because they are called from kernels: `injection`, `density_current`, `turbulence`, `mountain_waves`, `thermal`, `collision`, `hydro_const_bvfreq`, `hydro_const_theta`, and `sample_ellipse_cosine`.

Next, you'll need to create new send and recv MPI buffers that are created in CPU memory to easiy interoperate with the MPI library. To do this, you'll use the `realArrHost` `typedef` in `const.h`. 

```C++
realArrHost sendbuf_l_cpu;
realArrHost sendbuf_r_cpu;
realArrHost recvbuf_l_cpu;
realArrHost recvbuf_r_cpu;
```

Next, you need to allocate these in `init()` in a similar manner as the existing MPI buffers, but replacing `realArr` with `realArrHost`. 

Finally, you'll need to manage data movement to and from the CPU in the File I/O and in the MPI message exchanges.

For the File I/O, you'll need to use the `createHostCopy()` `Array` member function in the `ncmpi_put_*` routines, and you can use it in place before the `.data()` function.

For the MPI buffers, you'll need to use the `deep_copy_to()` `Array` member function....

Finally, you'll need to create new send and receive buffers in the x-direction halo exchange that are valid on the CPU using the realArrHost typdef in const.h. Then you'll need to copy the data to the CPU with the deep_copy_to function in Array class. See more information here:

https://github.com/mrnorman/YAKL

