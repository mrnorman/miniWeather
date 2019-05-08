# A Practical Introduction to GPU Refactoring in FORTRAN with Directives for Climate

This is intended to be an evolving document that covers the basics of what you should expect to do in a GPU refactoring and porting effort in a climate code. It will probably have some additions over time.

For climate codes, there are rarely large hot spots to be ported where the majority of the runtime is offloaded to an accelerator. Rather, when offloading a climate code, many different kernels must be created. This, coupled with the fact that the codes are traditionally in Fortran and that domain scientists are continually altering the code, means Fortran accelerator directives seem to be the most sensible approach. When approaching a GPU refactoring project, there are a few ideas that can guide the effort. The goal of this page is to overview those overarching goals / challenges for climate.

## Helpful Links:
* https://www.openacc.org 
* https://www.openacc.org/sites/default/files/inline-files/OpenACC%20API%202.6%20Reference%20Guide.pdf
* https://www.openmp.org
* https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf
* https://devblogs.nvidia.com/getting-started-openacc/

Also checkout the many YouTube videos on OpenACC and OpenMP4.5 tutorials.

## Quick-Running Testing Framework
The first thing you need is a quick-running testing framework, ideally a test or set of tests that can run on the order of 10 minutes. They need to be able to catch your most common bugs. One problem with testing the answers in climate is that it describes chaotic non-linear fluid dynamics where the answer changes rapidly from bit-level arithmetic changes (which you are very likely to introduce). My means of testing the answer is to run for a day at O0 optimization and then again at O3 optimization. This provides a baseline of the expected differences cause by bit-level changes after one day of simluation. Then, a three-way diff can be performed between the two baselines and the refactored simulation to determine if you are outside the expected envolope of answer changes. It's important to have these tests because finding bugs that could only have been caused by tens of lines of code changes is significantly easier and faster than finding bugs "in the wild" with a full-fledged debugger.

## Hardware / Software Setup
For a hardware / software setup, you'll need a compiler that implements the standard you want to use for your directives-based port. PGI, Cray, and GNU support OpenACC, and PGI is to my knowledge the most complete and performant implementation. Cray has moved their development resources to OpenMP 4.5+, I believe, and GNU's implementation is a work in progress still. Also, you can get PGI Community Edition for free on any workstation, while Cray and GNU's OpenACC-enabled branch are both harder to get access to. Also, PGI's community edition ships with its own MPI, NetCDF and Parallel-NetCDF libraries.

For OpenMP 4.5, GNU supports it for C/C++ only, IBM XL supports it for Fortran and C, and Cray supports it for Fortran and C. Intel supports the standard, but I wouldn't expect them to support GPU offload targets. I know from experience that the IBM XL implementation for Fortran does not play well with anything remotely modern, including derived types. 

Regarding hardware, there are a lot of cheap desktop-class GPUs available, but nearly all of them only work well in single precision and have terrible double precision performance. If you want double precision performance, you'll have to fork out money for the $3k Titan V or a Tesla-class workstation GPU. You can always look up your favorite GPU here (  https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units ) to see how it performs in single and double precision. Often, it's easiest just to do the GPU-specific development and testing on LCF machines that have GPUs in them already. But, admittedly, the compile time on these machines, especially with PGI, is prohibitively long for any significant development effort.

## Expose Threading in a Tightly-Nested Manner
First and foremost, you need to expose as much threading as possible to put on the accelerator. The classic 5 Simulated Years Per wallclock Day (SYPD) throughput constraint requires us to strong scale to the point that we have very little work available per node. This, in turn, means that we need every last ounce of threading we can get to put on a GPU in order to keep it busy enough to provide a computational benefit over CPUs. This usually means that we have to end up pushing loops down into subroutine calls.

This is often the most significant (and the most painful) aspect of GPU refactoring in climate. When you push looping into subroutines, you must then elevate all variables that persist outside that suborutine to include the extra dimension that was pushed in. Often, this is most easily done by redimensioning the variables first and then pushing the loop down into the routine. The reason is variable redimensioning can be done more incrementally than redimensioning everything at once. Then, you can perform a relatively small refactor, run some tests, and ensure you haven't introduced a bug.

I say to expose threading in a "tightly-nested" manner, and what that means is that you don't have anything between successive do loops or successive "end do"s.

```
!This is NOT tightly nested
do k = 1 , nlev
  if (levelActive(k)) then
    do j = 1 , ny
      do i = 1 , nx
      enddo
    enddo
  endif
enddo


!This IS tightly nested
!$acc parallel loop collapse(3)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      if (levelActive(k)) then
        ...
      endif
    enddo
  enddo
enddo
```

The most common reasons for loops to not be tightly nested are if-statements and integer logic such as storing indices or precomputing reused floating point operations. Sometimes there is other intermittent work. The solution to making something tightly-nested is to either split off the intermittent work into its own set of loops or to push that work (or if-statement) down inside the loops like the example above. There is a CPU-performance issue with pushing if-statements down into the inner-most loop because CPUs generally cannot cast these statements as vector operations like SSE or AVX. However, on the GPU, as long as the if-statement doesn't branch too much, you won't even notice it.

Why do this tightly nested? Because we're going to "collapse" the loops together into a single, large chunk of parallel threads that can be run easily on the GPU. There are cases when you don't have to collapse all the looping, but I'm not going to cover them here. For most climate codes, we don't always know a priori what the innermost loop is going to be, and we don't want the GPU port to only give performance for one particular choice of loop size. Therefore, it's most flexibly efficient on the GPU to just collapse all available looping together.

## Declare Private Variables
Next, it's important to understand when you need to declare private variables. A private variable is a variable for which each thread has its own private copy. You never have to declare scalar variables as private in OpenACC, as the compiler automatically assumes each thread needs its own copy. You only have to declare arrays as private. I'm not sure if this is also the case in OpenMP. Further, you only need to create a private copy of values that are written to. if the variable is read-only inside the loops, you can share that copy between threads. An example of a private variable is below:

```
!$acc parallel loop collapse(3) private(tmp)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      tmp(1) = a(i,j,k,1)*b(i,j,k,1)*c(1)
      tmp(2) = a(i,j,k,2)*b(i,j,k,2)*c(2)
      d(i,j,k) = tmp(1) + tmp(2)
    enddo
  enddo
enddo
```

In the example above, tmp needs to be private because it is written to and clearly each thread needs its own copy. However, "c" is not written to, so it can be shared. So, realistically, just look for scalars and small arrays that don't depend on the indices in your loops, that are on the left-hand side of the equals sign. Those are the ones that generally need to be private. 

Having a fast-running testing framework in place will help you find cases quickly where you forgot to make something private to each of the threads.

## Handle Race Conditions
There are many cases where we have race conditions in the threading. A race condition is a situation where threads are writing to the same memory location at the same time. The three most common examples of race conditions and the ways to handle them are below:

### Reductions

```
!A reduction is most easily handled when it is done across all dimensions of the loops
max_wave = 0
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      cs = sqrt(gamma*p(i,j,k)/rho(i,j,k))
      max_wave_loc = max( abs(u(i,j,k)) , abs(v(i,j,k)) , abs(w(i,j,k)) ) + cs
      max_wave = max( max_wave , max_wave_loc )
    enddo
  enddo
enddo

!You handle this in OpenACC and OpenMP with the reduction clause
max_wave = 0
!$acc parallel loop collapse(3) reduction(max:max_wave) private(cs,max_wave_loc)
max_wave = 0
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      cs = sqrt(gamma*p(i,j,k)/rho(i,j,k))
      max_wave_loc = max( abs(u(i,j,k)) , abs(v(i,j,k)) , abs(w(i,j,k)) ) + cs
      max_wave = max( max_wave , max_wave_loc )
    enddo
  enddo
enddo
```

Please note that the reduction clause in my experience only performs well when it is over all loop dimensions. Partial reductions seem to be more performant when handled with the "atomic" clause, which is what we'll look at next. Also, I did declare scalars as private in the code above. This isn't necessary in OpenACC, but it doesn't hurt anything either, and I think you may need to declare scalars as private in OpenMP4.5. I'm not sure at the moment.

### Atomic Accesses
There are also cases (like partial reductions) where parallel threads want to update the same memory location at the same time. To spot these, look for variables on the left-hand side of the equals sign that have some but not all of the loop indices. An example of this is below:

```
!Here is an example of a race condition of multiple threads writing to the same location at the same time that is not a reduction
do k = 1 , nlev
  u_column(k) = 0
enddo
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      u_column(k) = u_column(k) + u(i,j,k)
    enddo
  enddo
enddo

!This can be most easily (and performantly) handled with a simple atomic statement
!$acc parallel loop
do k = 1 , nlev
  u_column(k) = 0
enddo
!$acc parallel loop collapse(3) private(tmp)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      !You can only apply an atomic statement that looks like: a=a+b
      !Therefore, you need to precompute the RHS of the update before doing the atomic statement
      tmp = (u(i,j,k) - u0(k))*factor_xy
      !Now, you can apply the atomic directive
      !$acc atomic update
      u_column(k) = u_column(k) + tmp
    enddo
  enddo
enddo
```

On the Kepler K20x GPU on Titan, there were not hardware-based atomic operations in double precision, and so they were emulated in software. For this reason, double precision atomics were extremely slow on Titan. However, on Pascal and Volta GPUs (i.e., Summit), there is hardware support for double precition atomics, and in my experience, they perform extremely well.

### Prefix / Cumulative Sums (AKA "scans")
A prefix sum is the worst case of race condition, and they do happen in climate models fairly regularly. A prefix sum is where you need the intermittent sums as you progress through the indices of a variable. For example, in hydrostasis, the pressure at a level is an accumulation of mass above that column, and since you need the pressure at each level, you must save the intermittent values. Therefore, the order in which the summation occurs matters, making atomics inapplicable. You can spot prefix sums because you end up with "var(k) = var(k-1) + ..." for example. You have one index of a variable depending on another index of that same variable. This can be sneaky sometimes, as the dependency might be spread out over several temporary variables and thus a bit harder to spot. The atmospheric column physics routines have a number of occasions where prefix sums are computed.

I think prefix sums are most easily handled by saving all the work you can to a global temporary array, then performing the prefix sum in serial in the dimension being summed over, and then returning to parallel work afterward. You're essentially reducing the amount of serial data accesses and computations as much as possible in this method. There are cases where you can do something smarter, but they only apply to cases where prefix sums are computed over all dimensions of the kernel, which I don't think ever happens in climate that I've seen. Typically for us, prefix sums are only done vertically. See the code below as an example:

```
!Usually, prefix sums end up putting the vertical column as the innermost loop
do j = 1 , ny
  do i = 1 , nx
    hy_press(i,j,nlev) = 0
    do k = nlev-1 , 1 , -1
      hy_press(i,j,k) = hy_press(i,j,k+1) + [Lots of arithmetic]
    enddo
    do k = 1 , nlev
      otherstuff(i,j,k) = hy_press(i,j,k) + [stuff]
    enddo
  enddo
enddo


!First, we compute the arithmetic completely in parallel
!$acc parallel loop collapse(3)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      hy_temp(i,j,k) = [lots of arithmetic]
      if (k == nlev) hy_press(i,j,nlev) = 0
    enddo
  enddo
enddo
!Next, we do the prefix sum serially in the dimension summed
!minimizing the amount of serial work
!$acc parallel loop collapse(2)
do j = 1 , ny
  do i = 1 , nx
    !$acc loop seq
    do k = nlev-1 , 1 , -1
      hy_press(i,j,k) = hy_press(i,j,k+1) + hy_temp(i,j,k)
    enddo
  enddo
enddo
!Finally, we return to full parallelism for the rest of the work
!$acc parallel loop collapse(3)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      otherstuff(i,j,k) = hy_press(i,j,k) + [stuff]
      ...
    enddo
  enddo
enddo
```

## Optimizing / Managing Data Movement
This is probably the second-most cumbersome task. You need the data to be in the right place at the right time. Most of the reason the GPU is so fast compared to the CPU is the improved memory bandwidth on the GPU – the rate at which you can flood data into the processors to do computations. This is great, but it makes things complicated because the GPU has a separate memory than the CPU does. The CPU uses main system DRAM, while the GPU has its own faster RAM. You need to move data between the two, and it's very expensive to do so. Therefore, ideally, you want to keep data on the GPU for as long as possible before passing it back to the CPU. Often, this means that even kernels that give almost no speed-up on the GPU need to be ported anyway, solely because it's so expensive to pass that data back. 

There are two main things to focus on here. First, is the data on the GPU when the kernel is called? Second, am I keeping the data on the GPU between kernel calls?

I believe this is most easily handled by using "nested" data statements when using GPU directives like OpenACC or OpenMP 4.5+. In OpenACC, we have copyin(...), copyout(...), and copy(...). In OpenMP4.5, we have map(to:...), map(from:...), and map(tofrom:...). These are all written from the perspective of the GPU. Copyin() and map(to:) copy data into the GPU's RAM from the host DRAM before a block is entered. Copyout() and map(from:) copy data to the host DRAM from the GPU's RAM after a block is completed. Copy() and map(tofrom:) copy data into the GPU's RAM before the block and back to the host's RAM after the block is completed. However, there's something further. If the data already exists on the GPU, each of these directives is ignored. The full data statement sematics are as follows:
* create() / map(alloc:): If the data is not already allocated on the GPU, then allocate it on the GPU before the block begins and deallocate it from the GPU after the block ends.
* copyin() / map(to:): If the data is not already allocated on the GPU, then allocate it on the GPU and copy it from host memory to GPU memory before the block begins; then deallocate it from the GPU after the block ends. 
* copyout() / map(from:): If the data is not already allocated on the GPU, then allocate it on the GPU before the block begins; then copy it from GPU memory to host memory and deallocate it from the GPU after the block ends.
* copy() / map(tofrom:): If the data is not already allocated on the GPU, then allocate it on the GPU and copy it from host memory to GPU memory before the block begins; then copy it from GPU memory to host memory and deallocate it from the GPU after the block ends

The "If the data is not already allocate on the GPU" part is extremely useful because now you can "nest" your data statements. You can have data statements on each kernel to make sure the data is where it needs to be. And then, you can wrap the kernels in data statements to keep the data resident on the GPU between each of those kernels (because the per-kernel data statements are now ignored). Then, you can wrap those data statements to keep data on the GPU between subroutine calls, for instance. An example is below:

```
!Without this data statement, data will be shuffled to and from the GPU between each kernel
!However, with this, the per-kernel data statements are ignored, and data stays on the GPU
!$acc data copyout(a) copyin(d,c,e) copy(b)


!$acc parallel loop collapse(3) copyin(b,c) copyout(a)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      a(i,j,k) = b(i,j,k) + c(i,j,k)
    enddo
  enddo
enddo
!$acc parallel loop collapse(3) copy(b) copyin(d)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      b(i,j,k) = b(i,j,k) + d(i,j,k)
    enddo
  enddo
enddo
!$acc parallel loop collapse(3) copyin(b,c) copyout(a)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      a(i,j,k) = b(i,j,k) * c(i,j,k)
    enddo
  enddo
enddo

!$acc end data
```

### Managed Memory
Now, you don't have to worry about this if you use CUDA managed memory, and Managed Memory is a great option in my opinion, especially for large, complex codes with many hundreds of kernels and variables. The PGI compiler supports "-ta=nvidia,managed" to hijack all "allocate" calls in files compiled with this flag and replace them with cudaMallocManaged under the hood. PGI also enables a pool allocator by default for these allocates since they are very expensive. Managed Memory will use the CUDA run time to page your data back and forth for you when it is accessed on the CPU or GPU.

You can see a performance penalty because it waits until you access the data before it pages it, leading to large latencies for kernels that touch data that was last touched on the CPU. To fix this, you can use prefetching through the CUDA runtime. CUDA Fortran supports this, or you could potentially wrap a C call to the CUDA runtime as well. To do this, see the following example that combines OpenACC and CUDA Fortran (with the PGI compiler in this case):

```
  subroutine prefetch_r4_flat(a,n)
    implicit none
    real(4) :: a(n)
    integer :: n
#if defined(_OPENACC) && defined (_CUDA)
    !$acc host_data use_device(a)
    ierr = cudaMemPrefetchAsync( a , n , acc_get_device_num(acc_device_nvidia) , acc_get_cuda_stream(asyncid_loc+1) )
    !$acc end host_data
#endif
  end subroutine prefetch_r4_flat
  ```

## Managing Small Kernels and Memory Allocations
There is one final annoyance that's important to consider. In climate, since we strong scale significantly, we often end up in situations where we simply don't have much threading to work with. This is a big problem because it leads to small kernel runtimes, and kernels have significant overhead to launch on the GPU. The launch overhead is often between 5-10 microseconds. However, many of our kernels take less than 10 microseconds to complete! Therefore, we need a way to hide launch overheads. Further, many times, we have to allocate data on the GPU within subroutines. The CPU uses a pool allocator to manage allocations in main memory, and therefore, it's pretty cheap. However, cudaMalloc() on the GPU is very expensive (especially since our kernels take so little time).

The way to handle these situations is to do asynchronous kernel launches. What this does is launches all the kernels at once from the host so that the launch overheads are overlapped with kernel execuation. While many of our kernels are small, some are large, and this way, the larger kernels overlap with both the launch overheads with smaller kernels and with the GPU memory allocations. This is done with the "async(stream)" clause in OpenACC, where operations are synchronous within a stream and parallel between different streams. You specify a stream with an integer value. In OpenMP4.5, you can specify `depend(inout: mystream) nowait` on the kernels and data statements, where `mystream` is an integer value that will be translated to a CUDA stream under the hood in the OpenMP runtime:

```
!In this example, all data and kernel statements are sent to the GPU at once
!and the launch overheads and allocations overlap. The allocations will only overlap
!if there are previous kernels in stream "1" to overlap with, though


!You can only do asynchronous data statements with "enter data" and "exit data"
!$acc enter data copyin(b,c,e,d) async(1)

!$acc parallel loop collapse(3) copyin(b,c) copyout(a) async(1)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      a(i,j,k) = b(i,j,k) + c(i,j,k)
    enddo
  enddo
enddo

!$acc parallel loop collapse(3) copy(b) copyin(d) async(1)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      b(i,j,k) = b(i,j,k) + d(i,j,k)
    enddo
  enddo
enddo
!$acc parallel loop collapse(3) copyin(b,c) copyout(a) async(1)
do k = 1 , nlev
  do j = 1 , ny
    do i = 1 , nx
      a(i,j,k) = b(i,j,k) * c(i,j,k)
    enddo
  enddo
enddo

!$acc exit data copyout(a,b) async(1)


!This is a barrier that waits for all GPU work in stream "1" to complete
!$acc wait(1)
```

## Dealing with Function Calls
It is possible to call functions from GPU kernels, and this is handled with the !$acc routine directive. That being said, in my experience, the routine directive is the most poorly implemented aspect of OpenACC and the most likely place to run into bugs. I've had a lot of frustration with this directive. It works fine if it's just a single function call. However, nested function calls with routine directives gets hairy sometimes. Further, it has worked best for me when it's a function that works on a scalar value or on a small array that isn't threaded. When the routine directive has parallelism in it, sometimes it works well, and sometimes it doesn't. If you need to thread within a subroutine, I feel it's often more robust performance and bug-wise to push looping down into that routine and collapse it into a separate kernel. But feel free to experiment with this. Again, I've had more problems with the routine directive than about anything else in OpenACC.
