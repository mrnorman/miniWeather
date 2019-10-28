# miniWeather
A mini app simulating weather-like flows for training in parallelizing accelerated HPC architectures. Currently includes:
* MPI (C, Fortran, and C++)
* OpenACC Offload (C and Fortran)
* OpenMP Threading (C and Fortran)
* OpenMP Offload (C and Fortran)
* C++ Portability
  * CUDA-like approach
  * https://github.com/mrnorman/YAKL/wiki/CPlusPlus-Performance-Portability-For-OpenACC-and-OpenMP-Folks
  * C++ code works on CPU, Nvidia GPUs, and AMD GPUs

Author: [Matt Norman](https://mrnorman.github.io)

For detailed documentation, please see the [documentation/miniWeather_documentation.pdf](https://github.com/mrnorman/miniWeather/blob/master/documentation/miniWeather_documentation.pdf) file

## Basic Setup
```bash
git clone git@github.com:mrnorman/miniWeather.git
cd miniWeather
git submodule update --init
```

## Software Dependencies
* Parallel-netcdf: https://trac.mcs.anl.gov/projects/parallel-netcdf
  * This is a dependency for two reasons: (1) NetCDF files are easy to visualize and convenient to work with; (2) The users of this code shouldn't have to write their own parallel I/O.
* Ncview: http://meteora.ucsd.edu/~pierce/ncview_home_page.html
  * This is the easiest way to visualize NetCDF files.
* MPI
* For OpenACC: An OpenACC-capable compiler (PGI, Cray, GNU)
* For OpenMP: An OpenMP offload capable compiler (Cray, XL)
* For C++ portability, Nvidia's CUB and AMD's hipCUB and rocPRIM are already included as submodules

## MiniWeather Model Scaling Details
If you wnat to do scaling studies with miniWeather, this section will be important to make sure you're doing an apples-to-apples comparison. 

* `sim_time`: The `sim_time` parameter does not mean the wall time it takes to simulate but rather refers amount of model time simulated. As you increase `sim_time`, you should expect the walltime to increase linearly.
* `nx_glob, nz_glob`: As a rule, it's easiest if you always keep `nx_glob = nz_glob * 2` since the domain is always 20km x 10km in the x- and z-directions. As you increase `nx_glob` (and proportionally `nz_glob`) by some factor `f`, the time step automatically reduced by that same factor, `f`. Therefore, increasing `nx_glob` by 2x leads to 8x more work that needs to be done. Thus, with the same amount of parallelism, you should expect a 2x increase in `nx_glob` and `nz_glob` to increase the walltime by 8x (neglecting parallel overhead concerns).
  * More precisely, the time step is directly proportional to the minimum grid spacing. The x- and y-direction grid spacingsb are: `dx=20km/nx_glob` and `dz=10km/nz_glob`. So as you decrease the minimum grid spacing (by increasing `nx_glob` and/or `nz_glob`), you proportionally decrease the size of the time step and therefore proportionally increase the number of time steps you need to complete the simulation (thus proportionally increasing the expected walltime).
* The larger the problem size, `nx_glob` and `nz_glob`, the lower the relative parallel overheads will be. You can get to a point where there isn't enough work on the accelerator to keep it busy and / or enough local work to amortize parallel overheads. At this point, you'll need to increase the problem size to see better scaling. This is a typical [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law) situation.
