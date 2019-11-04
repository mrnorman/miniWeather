
#ifndef _YAKL_H_
#define _YAKL_H_

#include <iostream>
#include <algorithm>

#ifdef __USE_CUDA__
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #include <cub/cub.cuh>
#elif defined(__USE_HIP__)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#else
  #define YAKL_LAMBDA [&]
  #define YAKL_INLINE inline
#endif


namespace yakl {

  typedef unsigned int  uint;

  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr functorBufSize = 1024*128;
  extern void *functorBuffer;
  extern int vectorSize;



  inline void fence() {
    #ifdef __USE_CUDA__
      cudaDeviceSynchronize();
    #endif
    #ifdef __USE_HIP__
      hipDeviceSynchronize();
    #endif
  }



  inline void init(int vectorSize_in=128) {
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



  inline void finalize() {
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
    template <class F> __global__ void cudaKernelVal(int n1, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }
    template <class F> __global__ void cudaKernelVal(int n1, int n2, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2;
      if (i < nTot) {
        int i1, i2;
        unpackIndices( i , n1 , n2 , i1 , i2 );
        f( i1 , i2 );
      }
    }
    template <class F> __global__ void cudaKernelVal(int n1, int n2, int n3, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3;
      if (i < nTot) {
        int i1, i2, i3;
        unpackIndices( i , n1 , n2 , n3 , i1 , i2 , i3 );
        f( i1 , i2 , i3 );
      }
    }
    template <class F> __global__ void cudaKernelVal(int n1, int n2, int n3, int n4, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4;
      if (i < nTot) {
        int i1, i2, i3, i4;
        unpackIndices( i , n1 , n2 , n3 , n4 , i1 , i2 , i3 , i4 );
        f( i1 , i2 , i3 , i4 );
      }
    }
    template <class F> __global__ void cudaKernelVal(int n1, int n2, int n3, int n4, int n5, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5;
      if (i < nTot) {
        int i1, i2, i3, i4, i5;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , i1 , i2 , i3 , i4 , i5 );
        f( i1 , i2 , i3 , i4 , i5 );
      }
    }
    template <class F> __global__ void cudaKernelVal(int n1, int n2, int n3, int n4, int n5, int n6, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , i1 , i2 , i3 , i4 , i5 , i6 );
        f( i1 , i2 , i3 , i4 , i5 , i6 );
      }
    }
    template <class F> __global__ void cudaKernelVal(int n1, int n2, int n3, int n4, int n5, int n6, int n7, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6*n7;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6, i7;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , i1 , i2 , i3 , i4 , i5 , i6 , i7 );
        f( i1 , i2 , i3 , i4 , i5 , i6 , i7 );
      }
    }
    template <class F> __global__ void cudaKernelVal(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6*n7*n8;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6, i7, i8;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , i1 , i2 , i3 , i4 , i5 , i6 , i7 , i8 );
        f( i1 , i2 , i3 , i4 , i5 , i6 , i7 , i8 );
      }
    }


    template <class F> __global__ void cudaKernelRef(int n1, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }
    template <class F> __global__ void cudaKernelRef(int n1, int n2, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2;
      if (i < nTot) {
        int i1, i2;
        unpackIndices( i , n1 , n2 , i1 , i2 );
        f( i1 , i2 );
      }
    }
    template <class F> __global__ void cudaKernelRef(int n1, int n2, int n3, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3;
      if (i < nTot) {
        int i1, i2, i3;
        unpackIndices( i , n1 , n2 , n3 , i1 , i2 , i3 );
        f( i1 , i2 , i3 );
      }
    }
    template <class F> __global__ void cudaKernelRef(int n1, int n2, int n3, int n4, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4;
      if (i < nTot) {
        int i1, i2, i3, i4;
        unpackIndices( i , n1 , n2 , n3 , n4 , i1 , i2 , i3 , i4 );
        f( i1 , i2 , i3 , i4 );
      }
    }
    template <class F> __global__ void cudaKernelRef(int n1, int n2, int n3, int n4, int n5, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5;
      if (i < nTot) {
        int i1, i2, i3, i4, i5;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , i1 , i2 , i3 , i4 , i5 );
        f( i1 , i2 , i3 , i4 , i5 );
      }
    }
    template <class F> __global__ void cudaKernelRef(int n1, int n2, int n3, int n4, int n5, int n6, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , i1 , i2 , i3 , i4 , i5 , i6 );
        f( i1 , i2 , i3 , i4 , i5 , i6 );
      }
    }
    template <class F> __global__ void cudaKernelRef(int n1, int n2, int n3, int n4, int n5, int n6, int n7, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6*n7;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6, i7;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , i1 , i2 , i3 , i4 , i5 , i6 , i7 );
        f( i1 , i2 , i3 , i4 , i5 , i6 , i7 );
      }
    }
    template <class F> __global__ void cudaKernelRef(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, F const &f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6*n7*n8;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6, i7, i8;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , i1 , i2 , i3 , i4 , i5 , i6 , i7 , i8 );
        f( i1 , i2 , i3 , i4 , i5 , i6 , i7 , i8 );
      }
    }


    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , F const &f ) {
      cudaKernelVal <<< (uint) (n1-1)/vectorSize+1 , vectorSize >>> ( n1 , f );
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , F const &f ) {
      size_t nTot = n1*n2;
      cudaKernelVal <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , f );
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , F const &f ) {
      size_t nTot = n1*n2*n3;
      cudaKernelVal <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , f );
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , F const &f ) {
      size_t nTot = n1*n2*n3*n4;
      cudaKernelVal <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , f );
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5;
      cudaKernelVal <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , f );
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5*n6;
      cudaKernelVal <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , n6 , f );
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5*n6*n7;
      cudaKernelVal <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , n6 , n7 , f );
    }
    template<class F , typename std::enable_if< sizeof(F) <= 4000 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , int n8 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5*n6*n7*n8;
      cudaKernelVal <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , f );
    }


    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      cudaKernelRef <<< (uint) (n1-1)/vectorSize+1 , vectorSize >>> ( n1 , *fp );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      size_t nTot = n1*n2;
      cudaKernelRef <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , *fp );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      size_t nTot = n1*n2*n3;
      cudaKernelRef <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , *fp );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      size_t nTot = n1*n2*n3*n4;
      cudaKernelRef <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , *fp );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      size_t nTot = n1*n2*n3*n4*n5;
      cudaKernelRef <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , *fp );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      size_t nTot = n1*n2*n3*n4*n5*n6;
      cudaKernelRef <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , n6 , *fp );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      size_t nTot = n1*n2*n3*n4*n5*n6*n7;
      cudaKernelRef <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , n6 , n7 , *fp );
    }
    template<class F , typename std::enable_if< sizeof(F) >= 4001 , int >::type = 0> inline void parallel_for_cuda( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , int n8 , F const &f ) {
      F *fp = (F *) functorBuffer;
      cudaMemcpyAsync(fp,&f,sizeof(F),cudaMemcpyHostToDevice);
      size_t nTot = n1*n2*n3*n4*n5*n6*n7*n8;
      cudaKernelRef <<< (uint) (nTot-1)/vectorSize+1 , vectorSize >>> ( n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , *fp );
    }
  #endif


  #ifdef __USE_HIP__
    template <class F> __global__ void hipKernel(int n1, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n1) {
        f( i );
      }
    }
    template <class F> __global__ void hipKernel(int n1, int n2, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2;
      if (i < nTot) {
        int i1, i2;
        unpackIndices( i , n1 , n2 , i1 , i2 );
        f( i1 , i2 );
      }
    }
    template <class F> __global__ void hipKernel(int n1, int n2, int n3, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3;
      if (i < nTot) {
        int i1, i2, i3;
        unpackIndices( i , n1 , n2 , n3 , i1 , i2 , i3 );
        f( i1 , i2 , i3 );
      }
    }
    template <class F> __global__ void hipKernel(int n1, int n2, int n3, int n4, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4;
      if (i < nTot) {
        int i1, i2, i3, i4;
        unpackIndices( i , n1 , n2 , n3 , n4 , i1 , i2 , i3 , i4 );
        f( i1 , i2 , i3 , i4 );
      }
    }
    template <class F> __global__ void hipKernel(int n1, int n2, int n3, int n4, int n5, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5;
      if (i < nTot) {
        int i1, i2, i3, i4, i5;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , i1 , i2 , i3 , i4 , i5 );
        f( i1 , i2 , i3 , i4 , i5 );
      }
    }
    template <class F> __global__ void hipKernel(int n1, int n2, int n3, int n4, int n5, int n6, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , i1 , i2 , i3 , i4 , i5 , i6 );
        f( i1 , i2 , i3 , i4 , i5 , i6 );
      }
    }
    template <class F> __global__ void hipKernel(int n1, int n2, int n3, int n4, int n5, int n6, int n7, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6*n7;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6, i7;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , i1 , i2 , i3 , i4 , i5 , i6 , i7 );
        f( i1 , i2 , i3 , i4 , i5 , i6 , i7 );
      }
    }
    template <class F> __global__ void hipKernel(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, F f) {
      size_t i = blockIdx.x*blockDim.x + threadIdx.x;
      size_t nTot = n1*n2*n3*n4*n5*n6*n7*n8;
      if (i < nTot) {
        int i1, i2, i3, i4, i5, i6, i7, i8;
        unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , i1 , i2 , i3 , i4 , i5 , i6 , i7 , i8 );
        f( i1 , i2 , i3 , i4 , i5 , i6 , i7 , i8 );
      }
    }


    template<class F> inline void parallel_for_hip( int n1 , F const &f ) {
      hipLaunchKernelGGL( hipKernel , dim3((n1-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , f );
    }
    template<class F> inline void parallel_for_hip( int n1 , int n2 , F const &f ) {
      size_t nTot = n1*n2;
      hipLaunchKernelGGL( hipKernel , dim3((nTot-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , n2 , f );
    }
    template<class F> inline void parallel_for_hip( int n1 , int n2 , int n3 , F const &f ) {
      size_t nTot = n1*n2*n3;
      hipLaunchKernelGGL( hipKernel , dim3((nTot-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , n2 , n3 , f );
    }
    template<class F> inline void parallel_for_hip( int n1 , int n2 , int n3 , int n4 , F const &f ) {
      size_t nTot = n1*n2*n3*n4;
      hipLaunchKernelGGL( hipKernel , dim3((nTot-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , n2 , n3 , n4 , f );
    }
    template<class F> inline void parallel_for_hip( int n1 , int n2 , int n3 , int n4 , int n5 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5;
      hipLaunchKernelGGL( hipKernel , dim3((nTot-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , n2 , n3 , n4 , n5 , f );
    }
    template<class F> inline void parallel_for_hip( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5*n6;
      hipLaunchKernelGGL( hipKernel , dim3((nTot-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , n2 , n3 , n4 , n5 , n6 , f );
    }
    template<class F> inline void parallel_for_hip( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5*n6*n7;
      hipLaunchKernelGGL( hipKernel , dim3((nTot-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , n2 , n3 , n4 , n5 , n6 , n7 , f );
    }
    template<class F> inline void parallel_for_hip( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , int n8 , F const &f ) {
      size_t nTot = n1*n2*n3*n4*n5*n6*n7*n8;
      hipLaunchKernelGGL( hipKernel , dim3((nTot-1)/vectorSize+1) , dim3(vectorSize) , (std::uint32_t) 0 , (hipStream_t) 0 , n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , f );
    }
  #endif



  template <class F> inline void parallel_for_cpu_serial( int n1 , F const &f ) {
    for (int i=0; i<n1; i++) {
      f(i);
    }
  }
  template <class F> inline void parallel_for_cpu_serial( int n1 , int n2 , F const &f ) {
    size_t nTot = n1*n2;
    for (int i=0; i<nTot; i++) {
      int i1, i2;
      unpackIndices( i , n1 , n2 , i1 , i2 );
      f(i1,i2);
    }
  }
  template <class F> inline void parallel_for_cpu_serial( int n1 , int n2 , int n3 , F const &f ) {
    size_t nTot = n1*n2*n3;
    for (int i=0; i<nTot; i++) {
      int i1, i2, i3;
      unpackIndices( i , n1 , n2 , n3 , i1 , i2 , i3 );
      f(i1,i2,i3);
    }
  }
  template <class F> inline void parallel_for_cpu_serial( int n1 , int n2 , int n3 , int n4 , F const &f ) {
    size_t nTot = n1*n2*n3*n4;
    for (int i=0; i<nTot; i++) {
      int i1, i2, i3, i4;
      unpackIndices( i , n1 , n2 , n3 , n4 , i1 , i2 , i3 , i4 );
      f(i1,i2,i3,i4);
    }
  }
  template <class F> inline void parallel_for_cpu_serial( int n1 , int n2 , int n3 , int n4 , int n5 , F const &f ) {
    size_t nTot = n1*n2*n3*n4*n5;
    for (int i=0; i<nTot; i++) {
      int i1, i2, i3, i4, i5;
      unpackIndices( i , n1 , n2 , n3 , n4 , n5 , i1 , i2 , i3 , i4 , i5 );
      f(i1,i2,i3,i4,i5);
    }
  }
  template <class F> inline void parallel_for_cpu_serial( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , F const &f ) {
    size_t nTot = n1*n2*n3*n4*n5*n6;
    for (int i=0; i<nTot; i++) {
      int i1, i2, i3, i4, i5, i6;
      unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , i1 , i2 , i3 , i4 , i5 , i6 );
      f(i1,i2,i3,i4,i5,i6);
    }
  }
  template <class F> inline void parallel_for_cpu_serial( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , F const &f ) {
    size_t nTot = n1*n2*n3*n4*n5*n6*n7;
    for (int i=0; i<nTot; i++) {
      int i1, i2, i3, i4, i5, i6, i7;
      unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , i1 , i2 , i3 , i4 , i5 , i6 , i7 );
      f(i1,i2,i3,i4,i5,i6,i7);
    }
  }
  template <class F> inline void parallel_for_cpu_serial( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , int n8, F const &f ) {
    size_t nTot = n1*n2*n3*n4*n5*n6*n7*n8;
    for (int i=0; i<nTot; i++) {
      int i1, i2, i3, i4, i5, i6, i7, i8;
      unpackIndices( i , n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8, i1 , i2 , i3 , i4 , i5 , i6 , i7 , i8 );
      f(i1,i2,i3,i4,i5,i6,i7,i8);
    }
  }



  template <class F> inline void parallel_for( int n1 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , f );
    #else
      parallel_for_cpu_serial( n1 , f );
    #endif
  }
  template <class F> inline void parallel_for( int n1 , int n2 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , n2 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , n2 , f );
    #else
      parallel_for_cpu_serial( n1 , n2 , f );
    #endif
  }
  template <class F> inline void parallel_for( int n1 , int n2 , int n3 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , n2 , n3 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , n2 , n3 , f );
    #else
      parallel_for_cpu_serial( n1 , n2 , n3 , f );
    #endif
  }
  template <class F> inline void parallel_for( int n1 , int n2 , int n3 , int n4 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , n2 , n3 , n4 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , n2 , n3 , n4 , f );
    #else
      parallel_for_cpu_serial( n1 , n2 , n3 , n4 , f );
    #endif
  }
  template <class F> inline void parallel_for( int n1 , int n2 , int n3 , int n4 , int n5 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , n2 , n3 , n4 , n5 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , n2 , n3 , n4 , n5 , f );
    #else
      parallel_for_cpu_serial( n1 , n2 , n3 , n4 , n5 , f );
    #endif
  }
  template <class F> inline void parallel_for( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , n2 , n3 , n4 , n5 , n6 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , n2 , n3 , n4 , n5 , n6 , f );
    #else
      parallel_for_cpu_serial( n1 , n2 , n3 , n4 , n5 , n6 , f );
    #endif
  }
  template <class F> inline void parallel_for( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , n2 , n3 , n4 , n5 , n6 , n7 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , n2 , n3 , n4 , n5 , n6 , n7 , f );
    #else
      parallel_for_cpu_serial( n1 , n2 , n3 , n4 , n5 , n6 , n7 , f );
    #endif
  }
  template <class F> inline void parallel_for( int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , int n8 , F const &f ) {
    #ifdef __USE_CUDA__
      parallel_for_cuda( n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , f );
    #elif defined(__USE_HIP__)
      parallel_for_hip( n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , f );
    #else
      parallel_for_cpu_serial( n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , f );
    #endif
  }



  template <class F> inline void parallel_for( char const * str , int n1 , F const &f ) {
    parallel_for( n1 , f );
  }
  template <class F> inline void parallel_for( char const * str , int n1 , int n2 , F const &f ) {
    parallel_for( n1 , n2 , f );
  }
  template <class F> inline void parallel_for( char const * str , int n1 , int n2 , int n3 , F const &f ) {
    parallel_for( n1 , n2 , n3 , f );
  }
  template <class F> inline void parallel_for( char const * str , int n1 , int n2 , int n3 , int n4 , F const &f ) {
    parallel_for( n1 , n2 , n3 , n4 , f );
  }
  template <class F> inline void parallel_for( char const * str , int n1 , int n2 , int n3 , int n4 , int n5 , F const &f ) {
    parallel_for( n1 , n2 , n3 , n4 , n5 , f );
  }
  template <class F> inline void parallel_for( char const * str , int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , F const &f ) {
    parallel_for( n1 , n2 , n3 , n4 , n5 , n6 , f );
  }
  template <class F> inline void parallel_for( char const * str , int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , F const &f ) {
    parallel_for( n1 , n2 , n3 , n4 , n5 , n6 , n7 , f );
  }
  template <class F> inline void parallel_for( char const * str , int n1 , int n2 , int n3 , int n4 , int n5 , int n6 , int n7 , int n8 , F const &f ) {
    parallel_for( n1 , n2 , n3 , n4 , n5 , n6 , n7 , n8 , f );
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



  /*
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
  template <class FP> YAKL_INLINE void addAtomic(FP &x, FP const val) {
    #ifdef __USE_CUDA__
      atomicAdd(&x,val);
    #else
      x += val;
    #endif
  }

  template <class FP> YAKL_INLINE void minAtomic(FP &a, FP const b) {
    #ifdef __USE_CUDA__
      atomicMin(&a,b);
    #else
      a = a < b ? a : b;
    #endif
  }

  template <class FP> YAKL_INLINE void maxAtomic(FP &a, FP const b) {
    #ifdef __USE_CUDA__
      atomicMax(&a,b);
    #else
      a = a > b ? a : b;
    #endif
  }
  */

}


#endif

