
# Instructions for Petascale Institute on Blue Waters

Author: [Matt Norman](https://mrnorman.github.io)

**Table of Contents**
 * [Getting Started](#getting-started)
   + [ORNL Participants](#ornl-participants)
 * [Compiling the Code](#compiling-the-code)
 * [Running the Code and Viewing the Results](#running-the-code-and-viewing-the-results)
 * [Adding OpenACC](#adding-openacc)
   + [OpenACC Example](#openacc-example)
   + [Making the OpenACC Exercise Easier](#making-the-openacc-exercise-easier)
 * [Playing with the code](#playing-with-the-code)
   + [Changing the Problem Size](#changing-the-problem-size)
   + [Profiling](#profiling)
   + [Using Managed Memory](#using-managed-memory)
   + [Debugging with PGI's PCAST](#debugging-with-pgis-pcast)
   + [Debugging with cuda-memcheck](#debugging-with-cuda-memcheck)
   + [Debugging with valgrind](#debugging-with-valgrind)
 * [Other Resources](#other-resources)

## Getting Started
First, log in with:

```
ssh -Y [username]@bwbay.ncsa.illinois.edu
```

### ORNL Participants

ORNL by default blocks outgoing SSH, including to Blue Waters. To get around this, you have two options:

1. Use "corkscrew" (`apt-get intsall corkscrew` on Ubuntu), and use the ssh command

```
ssh -o ProxyCommand="corkscrew snowman.ornl.gov 3128 %h %p" -Y [username]@bwbay.ncsa.illinois.edu
```

2. ssh into `home.ccs.ornl.gov` using 2-factor authentication first, and then use the normal ssh command above.

Then clone this repo

```
git clone git@github.com:mrnorman/miniWeather.git
cd miniWeather
```

Choose either the `c` or `fortran` directories to work with, depending on your language preference.
All documentation is in [documentation/miniWeather_documentation.pdf](https://github.com/mrnorman/miniWeather/blob/master/documentation/miniWeather_documentation.pdf).

This code can be used to learn three different parallel programming aspects:
1. MPI task-level parallelism
2. OpenMP loop-level parallelism
3. OpenACC accelerator parallelism

For this particular context, we are focusing on the OpenACC parallelism, so the goal is to add your own OpenACC directives to `miniWeather_mpi.F90` or `miniWeather_mpi.cpp`. There are exmaples of how to do this with a loop from the code below in C and Fortran. You'll see there already exists a `miniWeather_mpi_openacc.F90` file, where the OpenACC directives have already been introduced. Please use it as a resource, but be aware that the data movement in that file has already been optimized, so the individual kernels no longer have data statements. Your code will probably look different as you add OpenACC directives yourself.

## Compiling the Code

To compile the code, you first need to create the correct environment with:

```
module swap PrgEnv-cray PrgEnv-pgi
module swap pgi pgi/18.7.0
module load cudatoolkit cray-parallel-netcdf
```

Use the `Makefile.bw` to compile the code:

```
cd miniWeather/[c | fortran]
make -f Makefile.bw
```

## Running the Code and Viewing the Results

To start an interactive 1-node job for an hour that contains a K20x GPU:

### On Tuesday afternoon, Use:
```
qsub -I -X -A bayr -lwalltime=01:00:00,nodes=1:ppn=16:xk
```

### On Wednesday, Use:
```
qsub -I -X -A bayr -lwalltime=01:00:00,nodes=1:ppn=16:xk
```

The `-X` flag is important if you want to view the NetCDF output file or profiler output from the Nvidia profile (see further down) because it enables X11 forwarding for the interactive job.

Next, setup the correct environment:

```
module swap PrgEnv-cray PrgEnv-pgi
module swap pgi pgi/18.7.0
module load cudatoolkit cray-parallel-netcdf ncview
```

Run the code:

```
aprun -n 1 ./miniWeather_mpi_openacc
```

OR

```
aprun -n 1 ./miniWeather_mpi
```

View the results

```
ncview output.nc
```

**IMPORTANT!**: Do not run `./miniWeather*` without `aprun`, as this will run it either on the login node or a service node. Those nodes are shared resources with limited resources, and you risk keeping others from compiling their code or even potentially killing other peoples' jobs if you cause an OOM on a service node. So please, be a good neighbor, and always use `aprun` for anything computationally intensive.

## Adding OpenACC

Again, for a fuller description, please see the [OpenACC presentation](https://github.com/mrnorman/miniWeather/blob/master/documentation/intro_to_openacc.pdf) and the [miniWeather documentation](https://github.com/mrnorman/miniWeather/blob/master/documentation/miniWeather_documentation.pdf). The code guides you where to put OpenACC directives with the following comment headers in `miniWeather_mpi.[F90 | cpp]`.

Fortran:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! TODO: THREAD ME
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

C:

```
/////////////////////////////////////////////////
// TODO: THREAD ME
/////////////////////////////////////////////////
```

### OpenACC Example

Here is an example of how to add directives to a loop with data statements.

#### Fortran:

```
do ll = 1 , NUM_VARS
  do k = 1 , nz
    do i = 1 , nx
      tend(i,k,ll) = -( flux(i+1,k,ll) - flux(i,k,ll) ) / dx
    enddo
  enddo
enddo
```

Becomes

```
!$acc parallel loop collapse(3) copyin(flux) copyout(tend)
do ll = 1 , NUM_VARS
  do k = 1 , nz
    do i = 1 , nx
      tend(i,k,ll) = -( flux(i+1,k,ll) - flux(i,k,ll) ) / dx
    enddo
  enddo
enddo
```

C:

```
for (ll=0; ll<NUM_VARS; ll++) {
  for (k=0; k<nz; k++) {
    for (i=0; i<nx; i++) {
      indt  = ll* nz   * nx    + k* nx    + i  ;
      indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
      indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
      tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
    }
  }
}
```

Becomes

```
#pragma acc parallel loop collapse(3) copyin(flux[NUM_VARS*(nz+1)*(nx+1)]) copyout(tend[NUM_VARS*nz*nx])
for (ll=0; ll<NUM_VARS; ll++) {
  for (k=0; k<nz; k++) {
    for (i=0; i<nx; i++) {
      indt  = ll* nz   * nx    + k* nx    + i  ;
      indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
      indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
      tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
    }
  }
}
```

### Making the OpenACC Exercise Easier

If you want to make things easier on yourself, especially in C, where the compiler does not generate data statements for you, I recommend using the PGI compiler with Managed Memory. The reason is that you no longer need to add your own data statements (e.g., you don't need any `copy()`, `copyin()`, or `copyout()` statements) because the CUDA runtime will move the data back and forth between CPU and GPU memory for you transparently under the hood. This means you can just focus on the `parallel loop` directive.

See the [Managed Memory](#using-managed-memory) section for more information on this.

## Playing with the code

### Changing the Problem Size

Find the `BEGIN USER-CONFIGURABLE PARAMETERS` in the source file, and change the number of grid points and the data specification according to the options in `documentation.pdf` in this repo. One interesting thing you can do is see how the OpenACC GPU efficiency changes as you change the problem size in terms of number of grid cells. The smaller the workload becomes, the worse the GPU performance should get because you begin to have too few threads to keep the device busy.

### Profiling

You can profile the code with:

```
aprun -n 1 nvprof --profile-child-processes -o %h.%p.nvvp ./miniWeather_mpi_openacc
```

and view the profile with:

```
nvvp [filename].nvvp
```

You will need X11 forwarding for this, though. To view a reduced text-only profile, use:

```
aprun -n 1 nvprof --profile-child-processes --print-gpu-summary ./miniWeather_mpi_openacc
```

### Using Managed Memory

The PGI compiler allows you to use Managed Memory instead of explicit data statements via the `-ta=nvidia,managed` flag. Try editing the Makefile to use Managed Memory, and see how the performance of the code changes. To do this, change the `ACCFLAGS` in `Makefile.bw` to:

```
ACCFLAGS := -ta=tesla,pinned,cc35,managed,ptxinfo -Minfo=accel 
```

### Debugging with PGI's PCAST

The PGI compiler also has a neat tool called PCAST, which automatically compares variables from redundatly executed CPU and GPU versions of the OpenACC code every time you move data from GPU memory to CPU memory (e.g., `update host(...)` or `copyout(...)` or the end of a block that has `copy(...)`). You can also control the size of the absolute or relative differences that will trigger terminal output. Change your `ACCFLAGS` in `Makefile.bw` to:

```
ACCFLAGS := -ta=tesla,pinned,cc35,autocompare,ptxinfo -Minfo=accel 
```

And recompile the code. For more options regarding the PCAST tool, see: https://www.pgroup.com/blogs/posts/pcast.htm

### Debugging with `cuda-memcheck`

Nvidia has a `valgrind`-esque tool called `cuda-memcheck`, which checks for invalid memory address errors in your GPU code, and can be quite handy in finding bugs. To use it, use the original `ACCFLAGS` specified in the repo's version of `Makefile.bw`, compile the code, and run:

```
aprun -n 1 cuda-memcheck ./miniWeather_mpi_openacc
```

### Debugging with `valgrind`

`valgrind` is a very useful CPU memory tool that checks for memory leaks and invalid accesses (among other things). To debug with valgrind on Blue Waters, you'll need to first load the module:

```
module load valgrind
```

Then, add the `-g` option to the `CFLAGS` in `Makefile.bw` so that valgrind can see the symbols and give you useful output. Then, simply run:

```
aprun -n 1 valgrind ./miniWeather_mpi
```

to see if you're committing any memory sins in your code. Keep in mind, this is for CPU code, not GPU code. In fact, with PGI 18.7, you'll find what appears to be a compiler bug in the C version, where valgrind complains about invalid reads in the main time stepping loop with the PGI compiler but does not with the GNU compiler. 

## Other Resources

I have a fairly involved document about the process you should expect when porting a new code to OpenACC from scratch. It's geared toward climate models, but I think you'll find it helpful even if you're in a different domain:

https://github.com/mrnorman/miniWeather/wiki/A-Practical-Introduction-to-GPU-Refactoring-in-Fortran-with-Directives-for-Climate

