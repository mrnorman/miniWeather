
#ifndef _SARRAY_H_
#define _SARRAY_H_

#include <iostream>
#include <iomanip>
#include "YAKL.h"

namespace yakl {

/*
  This is intended to be a simple, low-overhead class to do multi-dimensional arrays
  without pointer dereferencing. It supports indexing and cout only up to 3-D.

  It templates based on array dimension sizes, which conveniently allows overloaded
  functions in the TransformMatrices class.
*/

template <class T, unsigned int D0, unsigned int D1=1, unsigned int D2=1, unsigned int D3=1> class SArray {

protected:

  typedef unsigned int uint;

  T mutable data[D0*D1*D2];

public :

  YAKL_INLINE SArray() { }
  YAKL_INLINE SArray(SArray &&in) {
    for (uint i=0; i < D0*D1*D2*D3; i++) { data[i] = in.data[i]; }
  }
  YAKL_INLINE SArray(SArray const &in) {
    for (uint i=0; i < D0*D1*D2*D3; i++) { data[i] = in.data[i]; }
  }
  YAKL_INLINE SArray &operator=(SArray &&in) {
    for (uint i=0; i < D0*D1*D2*D3; i++) { data[i] = in.data[i]; }
    return *this;
  }
  YAKL_INLINE ~SArray() { }

  YAKL_INLINE T &operator()(uint const i0) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
    #endif
    return data[i0];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
    #endif
    return data[i0*D1 + i1];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
    #endif
    return data[i0*D1*D2 + i1*D2 + i2];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2, uint const i3) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
      if (i3>D3-1) { printf("i3 > D3-1"); exit(-1); }
    #endif
    return data[i0*D1*D2*D3 + i1*D2*D3 + i2*D3 + i3];
  }

  template <class I, uint E0> YAKL_INLINE SArray<T,E0> operator*(SArray<I,D0> const &rhs) {
    //This template could match either vector-vector or matrix-vector multiplication
    if ( (D1*D2*D3 == 1) ) {
      // Both 1-D Arrays --> Element-wise multiplication
      SArray<T,D0> ret;
      for (uint i=0; i<D0; i++) {
        ret.data[i] = data[i] * rhs.data[i];
      }
      return ret;
    } else {
      // Matrix-Vector multiplication
      SArray<T,D1> ret;
      for (uint j=0; j<D1; j++) {
        T tot = 0;
        for (uint i=0; i<D0; i++) {
          tot += (*this)(i,j) * rhs(i);
        }
        ret(j) = tot;
      }
      return ret;
    }
  }

  template <class I, uint E0> YAKL_INLINE SArray<T,E0,D1> operator*(SArray<I,E0,D0> const &rhs) {
    //This template matches Matrix-Matrix multiplication
    SArray<T,E0,D1> ret;
    for (uint j=0; j<E0; j++) {
      for (uint i=0; i<D1; i++) {
        T tot = 0;
        for (uint k=0; k<D0; k++) {
          tot += (*this)(k,i) * rhs(j,k);
        }
        ret(j,i) = tot;
      }
    }
    return ret;
  }

  YAKL_INLINE void operator=(T rhs) {
    //Scalar assignment
    for (uint i=0; i<D0*D1*D2*D3; i++) {
      data[i] = rhs;
    }
  }

  YAKL_INLINE T sum() {
    //Scalar division
    T sum = 0.;
    for (uint i=0; i<D0*D1*D2*D3; i++) {
      sum += data[i];
    }
    return sum;
  }

  YAKL_INLINE void operator/=(T rhs) {
    //Scalar division
    for (uint i=0; i<D0*D1*D2*D3; i++) {
      data[i] = data[i] / rhs;
    }
  }

  YAKL_INLINE void operator*=(T rhs) {
    //Scalar multiplication
    for (uint i=0; i<D0*D1*D2*D3; i++) {
      data[i] = data[i] * rhs;
    }
  }

  YAKL_INLINE SArray<T,D0,D1,D2> operator*(T rhs) {
    //Scalar multiplication
    SArray<T,D0,D1,D2> ret;
    for (uint i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] * rhs;
    }
    return ret;
  }

  YAKL_INLINE SArray<T,D0,D1,D2> operator/(T rhs) {
    //Scalar division
    SArray<T,D0,D1,D2> ret;
    for (uint i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] / rhs;
    }
    return ret;
  }

  inline friend std::ostream &operator<<(std::ostream& os, SArray const &v) {
    if (D1*D2*D3 == 1) {
      for (uint i=0; i<D0; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (D2*D3 == 1) {
      for (uint j=0; j<D1; j++) {
        for (uint i=0; i<D0; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else {
      for (uint i=0; i<D0*D1*D2*D3; i++) {
        os << std::setw(12) << v.data[i] << "\n";
      }
    }
    return os;
  }

};

}

#endif
