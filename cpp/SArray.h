
#ifndef _SARRAY_H_
#define _SARRAY_H_

#include <iostream>
#include <iomanip>
#include "YAKL.h"

/*
  This is intended to be a simple, low-overhead class to do multi-dimensional arrays
  without pointer dereferencing. It supports indexing and cout only up to 3-D.

  It templates based on array dimension sizes, which conveniently allows overloaded
  functions in the TransformMatrices class.
*/

template <class T, unsigned long D0, unsigned long D1=1, unsigned long D2=1, unsigned long D3=1> class SArray {

protected:

  typedef unsigned long ulong;

  T data[D0*D1*D2];

public :

  YAKL_INLINE SArray() { }
  // KOKKOS DOESN'T LIKE THESE
  // YAKL_INLINE SArray(SArray &&in) {
  //   for (int i=0; i < D0*D1*D2*D3; i++) { data[i] = in.data[i]; }
  // }
  // YAKL_INLINE SArray &operator=(SArray &&in) {
  //   for (int i=0; i < D0*D1*D2*D3; i++) { data[i] = in.data[i]; }
  //   return *this;
  // }
  YAKL_INLINE ~SArray() { }

  YAKL_INLINE T &operator()(ulong const i0)       {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
    #endif
    return data[i0];
  }
  YAKL_INLINE T &operator()(ulong const i0, ulong const i1)       {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
    #endif
    return data[i0*D1 + i1];
  }
  YAKL_INLINE T &operator()(ulong const i0, ulong const i1, ulong const i2)       {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
    #endif
    return data[i0*D1*D2 + i1*D2 + i2];
  }
  YAKL_INLINE T &operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3)       {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
      if (i3>D3-1) { printf("i3 > D3-1"); exit(-1); }
    #endif
    return data[i0*D1*D2*D3 + i1*D2*D3 + i2*D3 + i3];
  }

  YAKL_INLINE T  operator()(ulong const i0) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
    #endif
    return data[i0];
  }
  YAKL_INLINE T  operator()(ulong const i0, ulong const i1) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
    #endif
    return data[i0*D1 + i1];
  }
  YAKL_INLINE T  operator()(ulong const i0, ulong const i1, ulong const i2) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
    #endif
    return data[i0*D1*D2 + i1*D2 + i2];
  }
  YAKL_INLINE T  operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
      if (i3>D3-1) { printf("i3 > D3-1"); exit(-1); }
    #endif
    return data[i0*D1*D2*D3 + i1*D2*D3 + i2*D3 + i3];
  }

  template <class I, ulong E0> YAKL_INLINE SArray<T,E0> operator*(SArray<I,D0> const &rhs) {
    //This template could match either vector-vector or matrix-vector multiplication
    if ( (D1*D2*D3 == 1) ) {
      // Both 1-D Arrays --> Element-wise multiplication
      SArray<T,D0> ret;
      for (ulong i=0; i<D0; i++) {
        ret.data[i] = data[i] * rhs.data[i];
      }
      return ret;
    } else {
      // Matrix-Vector multiplication
      SArray<T,D1> ret;
      for (ulong j=0; j<D1; j++) {
        T tot = 0;
        for (ulong i=0; i<D0; i++) {
          tot += (*this)(i,j) * rhs(i);
        }
        ret(j) = tot;
      }
      return ret;
    }
  }

  template <class I, ulong E0> YAKL_INLINE SArray<T,E0,D1> operator*(SArray<I,E0,D0> const &rhs) {
    //This template matches Matrix-Matrix multiplication
    SArray<T,E0,D1> ret;
    for (ulong j=0; j<E0; j++) {
      for (ulong i=0; i<D1; i++) {
        T tot = 0;
        for (ulong k=0; k<D0; k++) {
          tot += (*this)(k,i) * rhs(j,k);
        }
        ret(j,i) = tot;
      }
    }
    return ret;
  }

  YAKL_INLINE void operator=(T rhs) {
    //Scalar assignment
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      data[i] = rhs;
    }
  }

  YAKL_INLINE T sum() {
    //Scalar division
    T sum = 0.;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      sum += data[i];
    }
    return sum;
  }

  YAKL_INLINE void operator/=(T rhs) {
    //Scalar division
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      data[i] = data[i] / rhs;
    }
  }

  YAKL_INLINE void operator*=(T rhs) {
    //Scalar multiplication
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      data[i] = data[i] * rhs;
    }
  }

  YAKL_INLINE SArray<T,D0,D1,D2> operator*(T rhs) {
    //Scalar multiplication
    SArray<T,D0,D1,D2> ret;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] * rhs;
    }
    return ret;
  }

  YAKL_INLINE SArray<T,D0,D1,D2> operator/(T rhs) {
    //Scalar division
    SArray<T,D0,D1,D2> ret;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] / rhs;
    }
    return ret;
  }

  inline friend std::ostream &operator<<(std::ostream& os, SArray const &v) {
    if (D1*D2*D3 == 1) {
      for (ulong i=0; i<D0; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (D2*D3 == 1) {
      for (ulong j=0; j<D1; j++) {
        for (ulong i=0; i<D0; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else {
      for (ulong i=0; i<D0*D1*D2*D3; i++) {
        os << std::setw(12) << v.data[i] << "\n";
      }
    }
    return os;
  }

};

#endif
