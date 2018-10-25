
#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>

#ifdef ARRAY_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif


/* Array<T>
   Implements an efficient multi-dimensional Array that allows basic matrix / vector arithmetic.
   Arrays up to 8 dimensions can be created through a few different constructors.
   You can also create an empty array and dimension it after-the-fact with Array.setup()

   This Array class is templated on the type of the data (float, double, int, etc)

   You can re-dimension an array with a subsequent call to Array.setup(), which deallocates the
   old data and creates a new internal array. The old data is destroyed.

   NOTE: This class was not developed with flexible dimension sizes in mind like std::vector. That
   setup does not perform well anyway, and it requires extra book keeping that I don't want to
   have to worry about. This class was designed for fixed-sized Arrays thorughout the object's
   lifetime. Redimensioning an array is expensive because it deallocates, allocates, and basically
   destroyed the previous data, requiring you to reset all of the data yourself. Arrays should only
   be redimensioned when you simply want to re-use the same object for a new application.

   You can access the array internals with accessor functions, but they are returned as const.

   I've added a number of global reductions for the Array class. I've tried to make them robust
   to integer types, but you'll have to look at the implementations to determine if they fit what
   you would expect for integer Array data. I always cast back to the class's templated type, meaning
   default rounding for integers in the mean, for instance.

   The only variable allocated on the heap with "new" is the Array's 1-D internall array "data". Since
   this structure is carefully handled internally, so long as you don't use "new", you'll never have
   a memory leak when using this class.

   This class is indexed by overloading the () operator, rather than the [] operator. This makes things
   a bit more flexible so you can create an array of Arrays if you wanted.

   All functions are templated, so you can index with whatever you want, but preferrably it would be
   integer-like to avoid undefined behavior.

   The internal indexing is always done with long and unsigned long so that you won't run into a case
   where the size is larger than the indexing can handle.

   For arithmetic, you can multiply by a constant, component-wise vectors, matvecs, and matmuls. 3-D or
   larger Arrays are multiplied component-wise. You can add a constant or another Array component-wise.

   By defining the ARRAY_DEBUG CPP variable at compile time, you can swich on "debug" mode, which is
   much slower than default mode without it. But it will catch all sorts of bugs you might otherwise
   introduce into the code by indexing out of bounds, with the wrong # dimensions, arithmetic on
   incompatible Array dimensions, etc.

   I've taken pains to make sure this class is (1) very fast in implementation and (2) able to be
   transferred to CUDA at a later date. Namely, there is no use of std:: outside of debug mode.
   It's more cumbersome up front, but CUDA doesn't recognize the standard template library yet
   (and maybe never will).

   Regarding fast implementation, everything but the data itself is part of the class data (i.e., on
   the stack and not on the heap), so it shouldn't create unnecessary cache misses. All accesses
   are into a 1-D array under the hood, so no pointer lookups. Integer arithmetic is reduced by
   creating "offsets" upon initialization for indexing. Further, the inline keyword is used everywhere
   to strongly encourage the compiler to do the right thing. Since debug functions are inlined
   when debug mode is off, they equate to an empty context and are optimized out.
*/


template <class T> class Array {

protected :

  typedef unsigned long ulong;

  int   ndims;
  long  lbounds [8];
  long  ubounds [8];
  ulong dimSizes[8];
  long  offsets [8];
  ulong totElems;
  T     *data;

  //Don't let the user have access to this. They might nullify an existing Array data, which
  //would make the pointer no longer accessible and thus create a memory leak.
  inline void nullify() {
    data = NULL;
    ndims = 0;
    totElems = 0;
    for (int i=0; i<8; i++) {
      dimSizes[i] = 0;
      lbounds [i] = 0;
      ubounds [i] = 0;
      offsets [i] = 0;
    }
  }

public :

  /* CONSTRUCTORS
     You can declare the array empty or with many dimensions
     Always nullify before beginning so that data == NULL upon init. This allows the
     setup() functions to keep from deallocating data upon initialization, since
     you don't know what "data" will be when the object is created.
  */
  Array() { nullify(); }
  //Define the dimension ranges using an array of upper bounds, assuming lower bounds to be zero
  Array(Array const &in) {
    nullify();
    setup(in);
  }
  template <class I> Array(I const d1) {
    nullify();
    setup(d1);
  }
  template <class I> Array(I const d1, I const d2) {
    nullify();
    setup(d1,d2);
  }
  template <class I> Array(I const d1, I const d2, I const d3) {
    nullify();
    setup(d1,d2,d3);
  }
  template <class I> Array(I const d1, I const d2, I const d3, I const d4) {
    nullify();
    setup(d1,d2,d3,d4);
  }
  template <class I> Array(I const d1, I const d2, I const d3, I const d4, I const d5) {
    nullify();
    setup(d1,d2,d3,d4,d5);
  }
  template <class I> Array(I const d1, I const d2, I const d3, I const d4, I const d5, I const d6) {
    nullify();
    setup(d1,d2,d3,d4,d5,d6);
  }
  template <class I> Array(I const d1, I const d2, I const d3, I const d4, I const d5, I const d6, I const d7) {
    nullify();
    setup(d1,d2,d3,d4,d5,d6,d7);
  }
  template <class I> Array(I const d1, I const d2, I const d3, I const d4, I const d5, I const d6, I const d7, I const d8) {
    nullify();
    setup(d1,d2,d3,d4,d5,d6,d7,d8);
  }
  //Define the dimension ranges using an array of upper bounds, assuming lower bounds to be zero
  template <class I> Array(I const ndims, I const dimSizes[]) {
    nullify();
    setup(ndims,dimSizes);
  }
  //Define the dimension ranges array of array, {{lbound,ubound},{lbound,ubound},...}
  template <class I> Array(I const ndims, I const bounds[][2]) {
    nullify();
    setup(ndims,bounds);
  }

  /*MOVE CONSTRUCTOR*/
  Array(Array &&in) {
    ndims    = in.ndims;
    totElems = in.totElems;
    for (int i=0; i < ndims; i++) {
      lbounds [i] = in.lbounds [i];
      ubounds [i] = in.ubounds [i];
      dimSizes[i] = in.dimSizes[i];
      offsets [i] = in.offsets [i];
    }
    data = in.data;
    in.data = NULL;
  }
  Array &operator=(Array &&rhs) {
    ndims    = rhs.ndims;
    totElems = rhs.totElems;
    for (int i=0; i < ndims; i++) {
      lbounds [i] = rhs.lbounds [i];
      ubounds [i] = rhs.ubounds [i];
      dimSizes[i] = rhs.dimSizes[i];
      offsets [i] = rhs.offsets [i];
    }
    data = rhs.data;
    rhs.data = NULL;
  }

  /* DESTRUCTOR
     Make sure the internal arrays are allocated before freeing them
  */
  ~Array() { finalize(); }

  /* SETUP FUNCTIONS
     Initialize the array with the given dimensions
  */
  inline void setup(Array const &in) {
    //If the buffer exists, and it's the right size, don't deallocate and reallocate
    if ( data != NULL && (this->totElems == in.totElems) )  {
      ndims    = in.ndims;
      for (int i=0; i < ndims; i++) {
        lbounds [i] = in.lbounds [i];
        ubounds [i] = in.ubounds [i];
        dimSizes[i] = in.dimSizes[i];
        offsets [i] = in.offsets [i];
      }
      for (ulong i=0; i < totElems; i++) {
        data[i] = in.data[i];
      }
    } else {
      finalize();
      ndims    = in.ndims;
      totElems = in.totElems;
      data = new T[totElems];
      for (int i=0; i < ndims; i++) {
        lbounds [i] = in.lbounds [i];
        ubounds [i] = in.ubounds [i];
        dimSizes[i] = in.dimSizes[i];
        offsets [i] = in.offsets [i];
      }
      for (ulong i=0; i < totElems; i++) {
        data[i] = in.data[i];
      }
    }
  }
  template <class I> inline void setup(I const d1) {
    ulong tmp[1];
    tmp[0] = d1;
    setup((ulong) 1,tmp);
  }
  template <class I> inline void setup(I const d1, I const d2) {
    ulong tmp[2];
    tmp[0] = d1;
    tmp[1] = d2;
    setup((ulong) 2,tmp);
  }
  template <class I> inline void setup(I const d1, I const d2, I const d3) {
    ulong tmp[3];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    setup((ulong) 3,tmp);
  }
  template <class I> inline void setup(I const d1, I const d2, I const d3, I const d4) {
    ulong tmp[4];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    setup((ulong) 4,tmp);
  }
  template <class I> inline void setup(I const d1, I const d2, I const d3, I const d4, I const d5) {
    ulong tmp[5];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    setup((ulong) 5,tmp);
  }
  template <class I> inline void setup(I const d1, I const d2, I const d3, I const d4, I const d5, I const d6) {
    ulong tmp[6];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    setup((ulong) 6,tmp);
  }
  template <class I> inline void setup(I const d1, I const d2, I const d3, I const d4, I const d5, I const d6, I const d7) {
    ulong tmp[7];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    setup((ulong) 7,tmp);
  }
  template <class I> inline void setup(I const d1, I const d2, I const d3, I const d4, I const d5, I const d6, I const d7, I const d8) {
    ulong tmp[8];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    tmp[7] = d8;
    setup((ulong) 8,tmp);
  }
  template <class I> inline void setup(I const ndims, I const dimSizes[]) {
    //Only deallocate and allocate if the buffer isn't yet allocate or the dimensions don't match
    if ( data == NULL || (! this->dimsMatch(ndims,dimSizes)) ) {
      finalize();
      this->ndims = ndims;
      totElems = 1;
      for (ulong i=0; i<ndims; i++) {
        this->lbounds [i] = 0;
        this->ubounds [i] = dimSizes[i]-1;
        this->dimSizes[i] = this->ubounds[i] - this->lbounds[i] + 1;
        totElems *= this->dimSizes[i];
      }
      offsets[ndims-1] = 1;
      for (int i=ndims-2; i>=0; i--) {
        offsets[i] = offsets[i+1] * dimSizes[i+1];
      }
      data = new T[totElems];
    }
  }
  template <class I> inline void setup(I const ndims, I const bounds[][2]) {
    finalize();
    this->ndims = ndims;
    totElems = 1;
    for (ulong i=0; i<ndims; i++) {
      this->lbounds [i] = bounds[i][0];
      this->ubounds [i] = bounds[i][1];
      this->dimSizes[i] = this->ubounds[i] - this->lbounds[i] + 1;
      totElems *= this->dimSizes[i];
    }
    offsets[ndims-1] = 1;
    for (int i=ndims-2; i>=0; i--) {
      offsets[i] = offsets[i+1] * dimSizes[i+1];
    }
    data = new T[totElems];
  }

  inline void finalize() {
    //Never "nullify()" until after the data is deallocated
    if (data != NULL) { delete[] data; nullify(); }
  }

  /* ARRAY INDEXERS (FORTRAN index ordering)
     Return the element at the given index (either read-only or read-write)
  */
  template <class I> inline T &operator()(I const i0) {
    this->check_dims(1,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0]);
    return data[ind];
  }
  template <class I> inline T &operator()(I const i0, I const i1) {
    this->check_dims(2,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1]);
    return data[ind];
  }
  template <class I> inline T &operator()(I const i0, I const i1, I const i2) {
    this->check_dims(3,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2]);
    return data[ind];
  }
  template <class I> inline T &operator()(I const i0, I const i1, I const i2, I const i3) {
    this->check_dims(4,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3]);
    return data[ind];
  }
  template <class I> inline T &operator()(I const i0, I const i1, I const i2, I const i3, I const i4) {
    this->check_dims(5,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4]);
    return data[ind];
  }
  template <class I> inline T &operator()(I const i0, I const i1, I const i2, I const i3, I const i4, I const i5) {
    this->check_dims(6,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    this->check_index(5,i5,lbounds[5],ubounds[5],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4])*offsets[4] +
                (i5-lbounds[5]);
    return data[ind];
  }
  template <class I> inline T &operator()(I const i0, I const i1, I const i2, I const i3, I const i4, I const i5, I const i6) {
    this->check_dims(7,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    this->check_index(5,i5,lbounds[5],ubounds[5],__FILE__,__LINE__);
    this->check_index(6,i6,lbounds[6],ubounds[6],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4])*offsets[4] +
                (i5-lbounds[5])*offsets[5] +
                (i6-lbounds[6]);
    return data[ind];
  }
  template <class I> inline T &operator()(I const i0, I const i1, I const i2, I const i3, I const i4, I const i5, I const i6, I const i7) {
    this->check_dims(8,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    this->check_index(5,i5,lbounds[5],ubounds[5],__FILE__,__LINE__);
    this->check_index(6,i6,lbounds[6],ubounds[6],__FILE__,__LINE__);
    this->check_index(7,i7,lbounds[7],ubounds[7],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4])*offsets[4] +
                (i5-lbounds[5])*offsets[5] +
                (i6-lbounds[6])*offsets[6] +
                (i7-lbounds[7]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0) const {
    this->check_dims(1,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0, I const i1) const {
    this->check_dims(2,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0, I const i1, I const i2) const {
    this->check_dims(3,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0, I const i1, I const i2, I const i3) const {
    this->check_dims(4,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0, I const i1, I const i2, I const i3, I const i4) const {
    this->check_dims(5,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0, I const i1, I const i2, I const i3, I const i4, I const i5) const {
    this->check_dims(6,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    this->check_index(5,i5,lbounds[5],ubounds[5],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4])*offsets[4] +
                (i5-lbounds[5]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0, I const i1, I const i2, I const i3, I const i4, I const i5, I const i6) const {
    this->check_dims(7,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    this->check_index(5,i5,lbounds[5],ubounds[5],__FILE__,__LINE__);
    this->check_index(6,i6,lbounds[6],ubounds[6],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4])*offsets[4] +
                (i5-lbounds[5])*offsets[5] +
                (i6-lbounds[6]);
    return data[ind];
  }
  template <class I> inline T operator()(I const i0, I const i1, I const i2, I const i3, I const i4, I const i5, I const i6, I const i7) const {
    this->check_dims(8,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,lbounds[0],ubounds[0],__FILE__,__LINE__);
    this->check_index(1,i1,lbounds[1],ubounds[1],__FILE__,__LINE__);
    this->check_index(2,i2,lbounds[2],ubounds[2],__FILE__,__LINE__);
    this->check_index(3,i3,lbounds[3],ubounds[3],__FILE__,__LINE__);
    this->check_index(4,i4,lbounds[4],ubounds[4],__FILE__,__LINE__);
    this->check_index(5,i5,lbounds[5],ubounds[5],__FILE__,__LINE__);
    this->check_index(6,i6,lbounds[6],ubounds[6],__FILE__,__LINE__);
    this->check_index(7,i7,lbounds[7],ubounds[7],__FILE__,__LINE__);
    ulong ind = (i0-lbounds[0])*offsets[0] +
                (i1-lbounds[1])*offsets[1] +
                (i2-lbounds[2])*offsets[2] +
                (i3-lbounds[3])*offsets[3] +
                (i4-lbounds[4])*offsets[4] +
                (i5-lbounds[5])*offsets[5] +
                (i6-lbounds[6])*offsets[6] +
                (i7-lbounds[7]);
    return data[ind];
  }

  inline void check_dims(int const ndims_called, int const ndims_actual, char const *file, int const line) const {
#ifdef ARRAY_DEBUG
    if (ndims_called != ndims_actual) {
      std::stringstream ss;
      ss << "Using " << ndims_called << " dimensions to index an Array with " << ndims_actual << " dimensions\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      throw std::out_of_range(ss.str());
    }
#endif
  }
  inline void check_index(int const dim, long const ind, long const lb, long const ub, char const *file, int const line) const {
#ifdef ARRAY_DEBUG
    if (ind < lb || ind > ub) {
      std::stringstream ss;
      ss << "Index " << dim << " of " << this->ndims << " out of bounds\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      ss << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw std::out_of_range(ss.str());
    }
#endif
  }

  /* OPERATOR+
     This will always be element-wise addition
  */
  template <class I> inline Array operator+(Array<I> const &rhs) {
#ifdef ARRAY_DEBUG
    if (this->totElems != rhs.totElems) {
      std::stringstream ss;
      ss << "Attempted element-wise addition between Arrays with incompatible lengths\n";
      ss << "File, Line: " << __FILE__ << ", " << __LINE__ << "\n";
      ss << "This size, rhs size:" << this->totElems << ", " << rhs.totElems << "\n";
      throw std::out_of_range(ss.str());
    }
#endif
    Array<T> ret(*this);
    for (ulong i=0; i<ret.totElems; i++) {
      ret.data[i] += rhs.data[i];
    }
    return ret;
  }
  //Add a scalar to the entire array
  template <class I> inline Array operator+(I const rhs) {
    Array<T> ret(*this);
    for (ulong i=0; i<ret.totElems; i++) {
      ret.data[i] += rhs;
    }
    return ret;
  }

  /* OPERATOR*
     1-D * 1-D: element-wise
     1-D * 2-D: INVALID
     2-D * 1-D: matrix-vector
     2-D * 2-D: matrix-matrix
     higher-dimensions: element-wise
  */
  template <class I> inline Array operator*(Array<I> const &rhs) {
    if ( (this->ndims == 1 && rhs.ndims == 1) || (this->ndims>2 && rhs.ndims>2) ) {
      //Element-wise multiplication
#ifdef ARRAY_DEBUG
    if (this->totElems != rhs.totElems) {
      std::stringstream ss;
      ss << "Attempted element-wise multiplication between Arrays with incompatible lengths\n";
      ss << "File, Line: " << __FILE__ << ", " << __LINE__ << "\n";
      ss << "This size, rhs size:" << this->totElems << ", " << rhs.totElems << "\n";
      throw std::out_of_range(ss.str());
    }
#endif
      Array<T> ret(*this);
      for (ulong i=0; i<ret.totElems; i++) {
        ret.data[i] *= rhs.data[i];
      }
      return ret;
    } else if (this->ndims == 2 && rhs.ndims == 1) {
      //Matrix-vector multiplication
#ifdef ARRAY_DEBUG
    if (this->dimSizes[0] != rhs.dimSizes[0]) {
      std::stringstream ss;
      ss << "Attempted matrix-vector multiplication between Arrays with incompatible dimensions\n";
      ss << "File, Line: " << __FILE__ << ", " << __LINE__ << "\n";
      ss << "this matrix dims(0,1), rhs dim: (" << this->dimSizes[0] << "," << this->dimSizes[1] << "), " << rhs.dimSizes[0] << "\n";
      throw std::out_of_range(ss.str());
    }
#endif
      Array<T> ret(this->dimSizes[1]);
      for (long j=0; j<this->dimSizes[1]; j++) {
        T tot = 0;
        for (long i=0; i<this->dimSizes[0]; i++) {
          tot += (*this)(i+this->lbounds[0],j+this->lbounds[1]) * rhs(i+rhs.lbounds[0]);
        }
        ret(j) = tot;
      }
      return ret;
    } else if (this->ndims == 2 && rhs.ndims == 2) {
      //Matrix-matrix multiplication
#ifdef ARRAY_DEBUG
    if (this->dimSizes[0] != rhs.dimSizes[1]) {
      std::stringstream ss;
      ss << "Attempted matrix-matrix multiplication between Arrays with incompatible dimensions\n";
      ss << "File, Line: " << __FILE__ << ", " << __LINE__ << "\n";
      ss << "this matrix dims, rhs matrix dims: (" << this->dimSizes[0] << "," << this->dimSizes[1] << ") , (" << rhs.dimSizes[0] << "," << rhs.dimSizes[1] << ")\n";
      throw std::out_of_range(ss.str());
    }
#endif
      Array<T> ret(rhs.dimSizes[0],this->dimSizes[1]);
      for (long j=0; j<rhs.dimSizes[0]; j++) {
        for (long i=0; i<this->dimSizes[1]; i++) {
          T tot = 0;
          for (long k=0; k<this->dimSizes[0]; k++) {
            tot += (*this)(k+this->lbounds[0],i+this->lbounds[1]) * rhs(j+rhs.lbounds[0],k+rhs.lbounds[1]);
          }
          ret(j,i) = tot;
        }
      }
      return ret;
    } else {
#ifdef ARRAY_DEBUG
      std::stringstream ss;
      ss << "Multiplying Arrays with incompatible dimensions\n";
      ss << "File, Line: " << __FILE__ << ", " << __LINE__ << "\n";
      throw std::out_of_range(ss.str());
#endif
    }
  }
  //Multiply by a scalar
  template <class I> inline Array operator*(I const rhs) {
    Array<T> ret(*this);
    for (ulong i=0; i<ret.totElems; i++) {
      ret.data[i] *= rhs;
    }
    return ret;
  }

  /* OPERATOR=
    Allow the user to set the entire Array to a single value */
  template <class I> inline void operator=(I const rhs) {
    for (ulong i=0; i < totElems; i++) {
      data[i] = rhs;
    }
  }
  /* Copy another Array's data to this one */
  inline void operator=(Array const &rhs) {
#ifdef ARRAY_DEBUG
    if (this->totElems != rhs.totElems) {
      std::stringstream ss;
      ss << "Attempted value-copy via operator= between Arrays with incompatible lengths\n";
      ss << "File, Line: " << __FILE__ << ", " << __LINE__ << "\n";
      ss << "This size, rhs size:" << this->totElems << ", " << rhs.totElems << "\n";
      throw std::out_of_range(ss.str());
    }
#endif
    for (ulong i=0; i < rhs.totElems; i++) {
      data[i] = rhs.data[i];
    }
  }
  /* Copy an array of values into this Array's data */
  template <class I> inline void operator=(I const *rhs) {
    for (ulong i=0; i<totElems; i++) {
      data[i] = rhs[i];
    }
  }

  /* RANDOM
    Sets the Array's data to a random initialization \in [range1,range2] */
  template <class I> inline void setRandom(I const range1, I const range2) {
    srand(time(NULL));
    for (ulong i=0; i<totElems; i++) {
      data[i] = range1 + static_cast <T> (rand()) /( static_cast <T> (RAND_MAX/(range2-range1)));
    }
  }

  /* REDUCTIONS
     Reductions over the Array. */
  inline T minval() const {
    T min = data[0];
    for (ulong i=1; i < totElems; i++) {
      if (data[i] < min) min = data[i];
    }
    return min;
  }
  inline T maxval() const {
    T max = data[0];
    for (ulong i=1; i < totElems; i++) {
      if (data[i] > max) max = data[i];
    }
    return max;
  }
  inline T maxabs() const {
    T max = fabs(data[0]);
    for (ulong i=1; i < totElems; i++) {
      if (fabs(data[i]) > max) max = fabs(data[i]);
    }
    return max;
  }
  inline T mean() const {
    T avg = 0.;
    for (ulong i=0; i < totElems; i++) {
      avg += data[i];
    }
    avg = avg / ((float) totElems);
    return (T) avg;
  }
  inline T product() const {
    T avg = 1.;
    for (ulong i=0; i < totElems; i++) {
      avg *= data[i];
    }
    return avg;
  }
  inline T sum() const {
    T sum = 0.;
    for (ulong i=0; i < totElems; i++) {
      sum += data[i];
    }
    return sum;
  }
  inline T variance() const {
    T mean = this->mean();
    T var = 0.;
    for (ulong i=0; i < totElems; i++) {
      var += (data[i] - mean)*(data[i] - mean);
    }
    return var;
  }
  inline T norm1() const {
    T norm = 0.;
    for (ulong i=0; i < totElems; i++) {
      norm += fabs(data[i]);
    }
    return norm;
  }
  inline T norm2() const {
    T norm = 0.;
    for (ulong i=0; i < totElems; i++) {
      norm += data[i]*data[i];
    }
    return norm;
  }
  inline T rms() const {
    T ret = 0.;
    for (ulong i=0; i < totElems; i++) {
      ret += data[i]*data[i];
    }
    return (T) sqrt( ret / ((float)totElems) );
  }

  /* NORMALIZERS */
  // mean of zero, standard deviation of 1
  inline Array<T> &normalizeNormal() {
    T avg = this->mean();
    T std = (T) sqrt(this->variance());
    if (std > 0) {
      for (ulong i=0; i<totElems; i++) {
        data[i] = ( data[i] - avg ) / std;
      }
    } else {
      (*this) = 0;
    }
    return *this;
  }
  // Linearly squash to the domain [-1,1]
  inline Array<T> &normalizeUnity() {
    T max = this->maxval();
    T min = this->minval();
    if (max-min > 0) {
      for (ulong i=0; i<totElems; i++) {
        data[i] = (data[i]-min) / (max-min) * 2 - 1;
      }
    } else {
      (*this) = 0;
    }
    return *this;
  }
  // Linearly squash to the domain [0,1]
  inline Array<T> &normalizeUnityPositive() {
    T max = this->maxval();
    T min = this->minval();
    if (max-min > 0) {
      for (ulong i=0; i<totElems; i++) {
        data[i] = (data[i]-min) / (max-min);
      }
    } else {
      (*this) = 0;
    }
    return *this;
  }

  /* COMPARISON */
  int dimsMatch(Array const &a) const {
    if (ndims != a.ndims) {
      return -1;
    }
    for (int i=0; i<ndims; i++) {
      if (dimSizes[i] != a.dimSizes[i]) {
        return -1;
      }
    }
    return 0;
  }
  template <class I> int dimsMatch(I const ndims, I const dimSizes[]) const {
    if (this->ndims != ndims) {
      return -1;
    }
    for (int i=0; i<ndims; i++) {
      if (this->dimSizes[i] != dimSizes[i]) {
        return -1;
      }
    }
    return 0;
  }

  /* ACCESSORS */
  inline int get_ndims() const {
    return ndims;
  }
  inline ulong get_totElems() const {
    return totElems;
  }
  inline ulong const *get_dimSizes() const {
    return dimSizes;
  }
  inline long const *get_lbounds() const {
    return lbounds;
  }
  inline long const *get_ubounds() const {
    return ubounds;
  }
  inline T *get_data() const {
    return data;
  }

  /* INFORM */
  inline void print_ndims() const {
    std::cout << "Number of Dimensions: " << ndims << "\n";
  }
  inline void print_totElems() const {
    std::cout << "Total Number of Elements: " << totElems << "\n";
  }
  inline void print_dimSizes() const {
    std::cout << "Dimension Sizes: ";
    for (int i=0; i<ndims; i++) {
      std::cout << dimSizes[i] << ", ";
    }
    std::cout << "\n";
  }
  inline void print_data() const {
    if (ndims == 1) {
      for (ulong i=0; i<dimSizes[0]; i++) {
        std::cout << std::setw(12) << (*this)(i) << "\n";
      }
    } else if (ndims == 2) {
      for (ulong j=0; j<dimSizes[0]; j++) {
        for (ulong i=0; i<dimSizes[1]; i++) {
          std::cout << std::setw(12) << (*this)(i,j) << " ";
        }
        std::cout << "\n";
      }
    } else if (ndims == 0) {
      std::cout << "Empty Array\n\n";
    } else {
      for (ulong i=0; i<totElems; i++) {
        std::cout << std::setw(12) << data[i] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  /* OPERATOR<<
     Print the array. If it's 2-D, print a pretty looking matrix */
  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
    os << "Number of Dimensions: " << v.ndims << "\n";
    os << "Total Number of Elements: " << v.totElems << "\n";
    os << "Dimension Sizes: ";
    for (int i=0; i<v.ndims; i++) {
      os << v.dimSizes[i] << ", ";
    }
    os << "\n";
    if (v.ndims == 1) {
      for (ulong i=v.lbounds[0]; i<=v.ubounds[0]; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (v.ndims == 2) {
      for (ulong j=v.lbounds[1]; j<=v.ubounds[1]; j++) {
        for (ulong i=v.lbounds[0]; i<=v.ubounds[0]; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else if (v.ndims == 0) {
      os << "Empty Array\n\n";
    } else {
      for (ulong i=0; i<v.totElems; i++) {
        os << v.data[i] << " ";
      }
      os << "\n";
    }
    os << "\n";
    return os;
  }

};

#endif
