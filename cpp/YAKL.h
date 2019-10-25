
#ifndef _YAKL_H_
#define _YAKL_H_

#include <iostream>
#include <algorithm>

#ifdef __USE_CUDA__
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE __host__ __device__
  #include <cub/cub.cuh>
#elif defined(__USE_HIP__)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE __host__ __device__
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#else
  #define YAKL_LAMBDA [=]
  #define YAKL_INLINE 
#endif


namespace yakl {


  int const memDevice = 1;
  int const memHost   = 2;


  void fence() {
    #ifdef __USE_CUDA__
      cudaDeviceSynchronize();
    #endif
    #ifdef __USE_HIP__
      hipDeviceSynchronize();
    #endif
  }


  typedef unsigned long ulong;
  typedef unsigned int  uint;

  
  int const functorBufSize = 1024*128;
  void *functorBuffer;
  int vectorSize = 128;


  void init(int vectorSize_in=128) {
    vectorSize = vectorSize_in;
    #if defined(__USE_CUDA__)
      cudaMalloc(&functorBuffer,functorBufSize);
    #endif
    #if defined(__USE_HIP__)
      int id;
      hipGetDevice(&id);
      hipDeviceProp_t props;
      hipGetDeviceProperties(&props,id);
      std::cout << props.name << std::endl;
    #endif
  }


  void finalize() {
    #if defined(__USE_CUDA__)
      cudaFree(functorBuffer);
    #endif
  }


  // Unpack 2D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int &i1, int &i2) {
    i1 = (iGlob/(n2))     ;
    i2 = (iGlob     ) % n2;
  }


  // Unpack 3D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int &i1, int &i2, int &i3) {
    i1 = (iGlob/(n3*n2))     ;
    i2 = (iGlob/(n3   )) % n2;
    i3 = (iGlob        ) % n3;
  }

  
  // Unpack 4D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int &i1, int &i2, int &i3, int &i4) {
    i1 = (iGlob/(n4*n3*n2))     ;
    i2 = (iGlob/(n4*n3   )) % n2;
    i3 = (iGlob/(n4      )) % n3;
    i4 = (iGlob           ) % n4;
  }

  
  // Unpack 5D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int &i1, int &i2, int &i3, int &i4, int &i5) {
    i1 = (iGlob/(n5*n4*n3*n2))     ;
    i2 = (iGlob/(n5*n4*n3   )) % n2;
    i3 = (iGlob/(n5*n4      )) % n3;
    i4 = (iGlob/(n5         )) % n4;
    i5 = (iGlob              ) % n5;
  }

  
  // Unpack 6D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6) {
    i1 = (iGlob/(n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n6*n5*n4      )) % n3;
    i4 = (iGlob/(n6*n5         )) % n4;
    i5 = (iGlob/(n6            )) % n5;
    i6 = (iGlob                 ) % n6;
  }

  
  // Unpack 7D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int n7, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7) {
    i1 = (iGlob/(n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n7*n6*n5         )) % n4;
    i5 = (iGlob/(n7*n6            )) % n5;
    i6 = (iGlob/(n7               )) % n6;
    i7 = (iGlob                    ) % n7;
  }

  
  // Unpack 8D indices
  YAKL_INLINE void unpackIndices(int iGlob, int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7, int &i8) {
    i1 = (iGlob/(n8*n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n8*n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n8*n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n8*n7*n6*n5         )) % n4;
    i5 = (iGlob/(n8*n7*n6            )) % n5;
    i6 = (iGlob/(n8*n7               )) % n6;
    i7 = (iGlob/(n8                  )) % n7;
    i8 = (iGlob                       ) % n8;
  }


  #ifdef __USE_CUDA__
    template <class F> __global__ void cudaKernelVal(ulong const nIter, F f) {
      ulong i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < nIter) {
        f( i );
      }
    }
    template <class F> __global__ void cudaKernelRef(ulong const nIter, F const &f) {
      ulong i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < nIter) {
        f( i );
      }
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> void parallel_for_cuda( int const nIter , F const &f ) {
      cudaKernelVal <<< (uint) (nIter-1)/vectorSize+1 , vectorSize >>> ( nIter , f );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> void parallel_for_cuda( int const nIter , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      cudaKernelRef <<< (uint) (nIter-1)/vectorSize+1 , vectorSize >>> ( nIter , *fp );
    }
  #endif


  #ifdef __USE_HIP__
    template <class F> __global__ void cudaKernelVal(ulong const nIter, F f) {
      ulong i = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
      if (i < nIter) {
        f( i );
      }
    }
    template<class F> void parallel_for_hip( int const nIter , F const &f ) {
      hipLaunchKernelGGL( cudaKernelVal , dim3((nIter-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , nIter , f );
    }
  #endif

  template <class F> void parallel_for_cpu_serial( int const nIter , F const &f ) {
    for (int i=0; i<nIter; i++) {
      f(i);
    }
  }


  template <class F> void parallel_for( int const nIter , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( nIter , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( nIter , f );
    #else
      parallel_for_cpu_serial( nIter , f );
    #endif
  }


  template <class F> void parallel_for( char const * str , int const nIter , F const &f ) {
    parallel_for( nIter , f );
  }


  template <class T, int myMem> class ParallelMin;
  template <class T, int myMem> class ParallelMax;
  template <class T, int myMem> class ParallelSum;
  #ifdef __USE_HIP__
    template <class T> class ParallelMin<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMin(int const nItems) {
        tmp  = NULL;
        nTmp = 0;
        hipMalloc(&rsltP,sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        hipcub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        hipMalloc(&tmp, nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      ~ParallelMin() {
        hipFree(rsltP);
        hipFree(tmp);
      }
      T operator() (T *data) {
        T rslt;
        hipcub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);    // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        hipcub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };
    template <class T> class ParallelMax<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMax(int const nItems) {
        tmp  = NULL;
        nTmp = 0;
        hipMalloc(&rsltP,sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        hipcub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        hipMalloc(&tmp, nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      ~ParallelMax() {
        hipFree(rsltP);
        hipFree(tmp);
      }
      T operator() (T *data) {
        T rslt;
        hipcub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        hipcub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };
    template <class T> class ParallelSum<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelSum(int const nItems) {
        tmp  = NULL;
        nTmp = 0;
        hipMalloc(&rsltP,sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        hipcub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        hipMalloc(&tmp, nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      ~ParallelSum() {
        hipFree(rsltP);
        hipFree(tmp);
      }
      T operator() (T *data) {
        T rslt;
        hipcub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        hipMemcpyAsync(&rslt,rsltP,sizeof(T),hipMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        hipcub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };
  #elif defined(__USE_CUDA__)
    template <class T> class ParallelMin<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMin(int const nItems) {
        tmp  = NULL;
        nTmp = 0;
        cudaMalloc(&rsltP,sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        cub::DeviceReduce::Min(tmp, nTmp, rsltP , rsltP , nItems );
        cudaMalloc(&tmp, nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      ~ParallelMin() {
        cudaFree(rsltP);
        cudaFree(tmp);
      }
      T operator() (T *data) {
        T rslt;
        cub::DeviceReduce::Min(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        cub::DeviceReduce::Min(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };
    template <class T> class ParallelMax<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelMax(int const nItems) {
        tmp  = NULL;
        nTmp = 0;
        cudaMalloc(&rsltP,sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        cub::DeviceReduce::Max(tmp, nTmp, rsltP , rsltP , nItems );
        cudaMalloc(&tmp, nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      ~ParallelMax() {
        cudaFree(rsltP);
        cudaFree(tmp);
      }
      T operator() (T *data) {
        T rslt;
        cub::DeviceReduce::Max(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        cub::DeviceReduce::Max(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };
    template <class T> class ParallelSum<T,memDevice> {
      void   *tmp;   // Temporary storage
      size_t nTmp;   // Size of temporary storage
      int    nItems; // Number of items in the array that will be reduced
      T      *rsltP; // Device pointer for reduction result
      public:
      ParallelSum(int const nItems) {
        tmp  = NULL;
        nTmp = 0;
        cudaMalloc(&rsltP,sizeof(T)); // Allocate device pointer for result
        // Get the amount of temporary storage needed (call with NULL storage pointer)
        cub::DeviceReduce::Sum(tmp, nTmp, rsltP , rsltP , nItems );
        cudaMalloc(&tmp, nTmp);       // Allocate temporary storage
        this->nItems = nItems;
      }
      ~ParallelSum() {
        cudaFree(rsltP);
        cudaFree(tmp);
      }
      T operator() (T *data) {
        T rslt;
        cub::DeviceReduce::Sum(tmp, nTmp, data , rsltP , nItems , 0 ); // Compute the reduction
        cudaMemcpyAsync(&rslt,rsltP,sizeof(T),cudaMemcpyDeviceToHost,0);       // Copy result to host
        fence();
        return rslt;
      }
      void deviceReduce(T *data, T *devP) {
        cub::DeviceReduce::Sum(tmp, nTmp, data , devP , nItems , 0 ); // Compute the reduction
      }
    };
  #else
    template <class T> class ParallelMin<T,memDevice> {
      int    nItems; // Number of items in the array that will be reduced
      public:
      ParallelMin(int const nItems) {
        this->nItems = nItems;
      }
      ~ParallelMin() {
      }
      T operator() (T *data) {
        T rslt = data[0];
        for (int i=1; i<nItems; i++) {
          rslt = data[i] < rslt ? data[i] : rslt;
        }
        return rslt;
      }
      void deviceReduce(T *data, T *rslt) {
        *(rslt) = data[0];
        for (int i=1; i<nItems; i++) {
          *(rslt) = data[i] < *(rslt) ? data[i] : rslt;
        }
      }
    };
    template <class T> class ParallelMax<T,memDevice> {
      int    nItems; // Number of items in the array that will be reduced
      public:
      ParallelMax(int const nItems) {
        this->nItems = nItems;
      }
      ~ParallelMax() {
      }
      T operator() (T *data) {
        T rslt = data[0];
        for (int i=1; i<nItems; i++) {
          rslt = data[i] > rslt ? data[i] : rslt;
        }
        return rslt;
      }
      void deviceReduce(T *data, T *rslt) {
        *(rslt) = data[0];
        for (int i=1; i<nItems; i++) {
          *(rslt) = data[i] > *(rslt) ? data[i] : rslt;
        }
      }
    };
    template <class T> class ParallelSum<T,memDevice> {
      int    nItems; // Number of items in the array that will be reduced
      public:
      ParallelSum(int const nItems) {
        this->nItems = nItems;
      }
      ~ParallelSum() {
      }
      T operator() (T *data) {
        T rslt = data[0];
        for (int i=1; i<nItems; i++) {
          rslt += data[i];
        }
        return rslt;
      }
      void deviceReduce(T *data, T *rslt) {
        *(rslt) = data[0];
        for (int i=1; i<nItems; i++) {
          *(rslt) += data[i];
        }
      }
    };
  #endif
  template <class T> class ParallelMin<T,memHost> {
    int    nItems; // Number of items in the array that will be reduced
    public:
    ParallelMin(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMin() {
    }
    T operator() (T *data) {
      T rslt = data[0];
      for (int i=1; i<nItems; i++) {
        rslt = data[i] < rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *rslt) {
      *(rslt) = data[0];
      for (int i=1; i<nItems; i++) {
        *(rslt) = data[i] < *(rslt) ? data[i] : rslt;
      }
    }
  };
  template <class T> class ParallelMax<T,memHost> {
    int    nItems; // Number of items in the array that will be reduced
    public:
    ParallelMax(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelMax() {
    }
    T operator() (T *data) {
      T rslt = data[0];
      for (int i=1; i<nItems; i++) {
        rslt = data[i] > rslt ? data[i] : rslt;
      }
      return rslt;
    }
    void deviceReduce(T *data, T *rslt) {
      *(rslt) = data[0];
      for (int i=1; i<nItems; i++) {
        *(rslt) = data[i] > *(rslt) ? data[i] : rslt;
      }
    }
  };
  template <class T> class ParallelSum<T,memHost> {
    int    nItems; // Number of items in the array that will be reduced
    public:
    ParallelSum(int const nItems) {
      this->nItems = nItems;
    }
    ~ParallelSum() {
    }
    T operator() (T *data) {
      T rslt = data[0];
      for (int i=1; i<nItems; i++) {
        rslt += data[i];
      }
      return rslt;
    }
    void deviceReduce(T *data, T *rslt) {
      *(rslt) = data[0];
      for (int i=1; i<nItems; i++) {
        *(rslt) += data[i];
      }
    }
  };


  #ifdef __USE_CUDA__
    __device__ __forceinline__ void atomicMin(float *address , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(*address);
      newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMin(double *address , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(*address);
      newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(float *address , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(*address);
      newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(double *address , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(*address);
      newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      }
    }
  #endif
  template <class FP> inline YAKL_INLINE void addAtomic(FP &x, FP const val) {
    #ifdef __USE_CUDA__
      atomicAdd(&x,val);
    #else
      x += val;
    #endif
  }

  template <class FP> inline YAKL_INLINE void minAtomic(FP &a, FP const b) {
    #ifdef __USE_CUDA__
      atomicMin(&a,b);
    #else
      a = a < b ? a : b;
    #endif
  }

  template <class FP> inline YAKL_INLINE void maxAtomic(FP &a, FP const b) {
    #ifdef __USE_CUDA__
      atomicMax(&a,b);
    #else
      a = a > b ? a : b;
    #endif
  }

}


#endif

