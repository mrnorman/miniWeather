
#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>
#include "stdlib.h"
#include "YAKL.h"

#ifdef ARRAY_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif

namespace yakl {


/* Array<T>
Multi-dimensional array with functor indexing up to eight dimensions.
*/

template <class T, int myMem> class Array {

  public :

  size_t offsets  [8];  // Precomputed dimension offsets for efficient data access into a 1-D pointer
  size_t dimension[8];  // Sizes of the 8 possible dimensions
  T      * myData;      // Pointer to the flattened internal data
  int    rank;          // Number of dimensions
  size_t totElems;      // Total number of elements in this Array
  int    * refCount;    // Pointer shared by multiple copies of this Array to keep track of allcation / free
  #ifdef ARRAY_DEBUG
    std::string myname; // Label for debug printing. Only stored if debugging is turned on
  #endif


  // Start off all constructors making sure the pointers are null
  YAKL_INLINE void nullify() {
    myData   = nullptr;
    refCount = nullptr;
    rank = 0;
    totElems = 0;
    for (int i=0; i<8; i++) {
      dimension[i] = 0;
      offsets  [i] = 0;
    }
  }

  /* CONSTRUCTORS
  You can declare the array empty or with up to 8 dimensions
  Like kokkos, you need to give a label for the array for debug printing
  Always nullify before beginning so that myData == nullptr upon init. This allows the
  setup() functions to keep from deallocating myData upon initialization, since
  you don't know what "myData" will be when the object is created.
  */
  YAKL_INLINE Array() {
    nullify();
  }
  YAKL_INLINE Array(char const * label) {
    nullify();
    #ifdef ARRAY_DEBUG
      myname = std::string(label);
    #endif
  }
  //Define the dimension ranges using an array of upper bounds, assuming lower bounds to be zero
  Array(char const * label, size_t const d1) {
    nullify();
    setup(label,d1);
  }
  Array(char const * label, size_t const d1, size_t const d2) {
    nullify();
    setup(label,d1,d2);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3) {
    nullify();
    setup(label,d1,d2,d3);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4) {
    nullify();
    setup(label,d1,d2,d3,d4);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5) {
    nullify();
    setup(label,d1,d2,d3,d4,d5);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6) {
    nullify();
    setup(label,d1,d2,d3,d4,d5,d6);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7) {
    nullify();
    setup(label,d1,d2,d3,d4,d5,d6,d7);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7, size_t const d8) {
    nullify();
    setup(label,d1,d2,d3,d4,d5,d6,d7,d8);
  }


  /*
  COPY CONSTRUCTORS / FUNCTIONS
  This shares the pointers with another Array and increments the refCounter
  */
  YAKL_INLINE Array(Array const &rhs) {
    nullify();
    rank     = rhs.rank;
    totElems = rhs.totElems;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    (*refCount)++;
  }


  Array & operator=(Array const &rhs) {
    if (this == &rhs) {
      return *this;
    }
    deallocate();
    rank     = rhs.rank;
    totElems = rhs.totElems;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    (*refCount)++;

    return *this;
  }


  /*
  MOVE CONSTRUCTORS
  This straight up steals the pointers form the rhs and sets them to null.
  */
  YAKL_INLINE Array(Array &&rhs) {
    nullify();
    rank     = rhs.rank;
    totElems = rhs.totElems;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;
  }


  Array& operator=(Array &&rhs) {
    if (this == &rhs) {
      return *this;
    }
    deallocate();
    rank     = rhs.rank;
    totElems = rhs.totElems;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;

    return *this;
  }


  /*
  DESTRUCTOR
  Decrement the refCounter, and if it's zero, deallocate and nullify.  
  */
  YAKL_INLINE ~Array() {
    deallocate();
  }


  /* SETUP FUNCTIONS
  Initialize the array with the given dimensions
  */
  inline void setup(char const * label, size_t const d1) {
    size_t tmp[1];
    tmp[0] = d1;
    setup_arr(label, (size_t) 1,tmp);
  }
  inline void setup(char const * label, size_t const d1, size_t const d2) {
    size_t tmp[2];
    tmp[0] = d1;
    tmp[1] = d2;
    setup_arr(label, (size_t) 2,tmp);
  }
  inline void setup(char const * label, size_t const d1, size_t const d2, size_t const d3) {
    size_t tmp[3];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    setup_arr(label, (size_t) 3,tmp);
  }
  inline void setup(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4) {
    size_t tmp[4];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    setup_arr(label, (size_t) 4,tmp);
  }
  inline void setup(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5) {
    size_t tmp[5];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    setup_arr(label, (size_t) 5,tmp);
  }
  inline void setup(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6) {
    size_t tmp[6];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    setup_arr(label, (size_t) 6,tmp);
  }
  inline void setup(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7) {
    size_t tmp[7];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    setup_arr(label, (size_t) 7,tmp);
  }
  inline void setup(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7, size_t const d8) {
    size_t tmp[8];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    tmp[7] = d8;
    setup_arr(label, (size_t) 8,tmp);
  }
  inline void setup_arr(char const * label, size_t const rank, size_t const dimension[]) {
    #ifdef ARRAY_DEBUG
      myname = std::string(label);
    #endif

    deallocate();

    // Setup this Array with the given number of dimensions and dimension sizes
    this->rank = rank;
    totElems = 1;
    for (size_t i=0; i<rank; i++) {
      this->dimension[i] = dimension[i];
      totElems *= this->dimension[i];
    }
    offsets[rank-1] = 1;
    for (int i=rank-2; i>=0; i--) {
      offsets[i] = offsets[i+1] * dimension[i+1];
    }
    allocate();
  }


  inline void allocate() {
    refCount = new int;
    *refCount = 1;
    if (myMem == memDevice) {
      #ifdef __USE_CUDA__
        cudaMalloc(&myData,totElems*sizeof(T));
      #elif defined(__USE_HIP__)
        hipMalloc(&myData,totElems*sizeof(T));
      #endif
    } else {
      myData = new T[totElems];
    }
  }


  inline void deallocate() {
    if (refCount != nullptr) {
      (*refCount)--;

      if (*refCount == 0) {
        delete refCount;
        refCount = nullptr;
        if (myMem == memDevice) {
          #ifdef __USE_CUDA__
            cudaFree(myData);
          #elif defined(__USE_HIP__)
            hipFree(myData);
          #endif
        } else {
          delete[] myData;
        }
        myData = nullptr;
      }

    }
  }


  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  YAKL_INLINE T &operator()(size_t const i0) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(1,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(2,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(3,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(4,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(5,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4, size_t const i5) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(6,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4, size_t const i5, size_t const i6) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(7,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,0,dimension[6]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4, size_t const i5, size_t const i6, size_t const i7) const {
    #ifdef ARRAY_DEBUG
      this->check_dims(8,rank,__FILE__,__LINE__);
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,0,dimension[6]-1,__FILE__,__LINE__);
      this->check_index(7,i7,0,dimension[7]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6*offsets[6] + i7;
    return myData[ind];
  }


  inline void check_dims(int const rank_called, int const rank_actual, char const *file, int const line) const {
    #ifdef ARRAY_DEBUG
    if (rank_called != rank_actual) {
      std::stringstream ss;
      ss << "For Array labeled: " << myname << "\n";
      ss << "Using " << rank_called << " dimensions to index an Array with " << rank_actual << " dimensions\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      throw std::out_of_range(ss.str());
    }
    #endif
  }
  inline void check_index(int const dim, long const ind, long const lb, long const ub, char const *file, int const line) const {
    #ifdef ARRAY_DEBUG
    if (ind < lb || ind > ub) {
      std::stringstream ss;
      ss << "For Array labeled: " << myname << "\n";
      ss << "Index " << dim << " of " << this->rank << " out of bounds\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      ss << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw std::out_of_range(ss.str());
    }
    #endif
  }


  YAKL_INLINE T sum() const {
    T sum = 0.;
    for (size_t i=0; i < totElems; i++) {
      sum += myData[i];
    }
    return sum;
  }


  inline Array<T,memHost> createHostCopy() {
    Array<T,memHost> ret;
    #ifdef ARRAY_DEBUG
      ret.setup_arr( myname.c_str() , rank , dimension );
    #else
      ret.setup_arr( ""             , rank , dimension );
    #endif
    if (myMem == memHost) {
      for (int i=0; i<totElems; i++) {
        ret.myData[i] = myData[i];
      }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems*sizeof(T),cudaMemcpyDeviceToHost,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems*sizeof(T),hipMemcpyDeviceToHost,0);
        hipDeviceSynchronize();
      #endif
    }
    return ret;
  }


  inline Array<T,memDevice> createDeviceCopy() {
    Array<T,memDevice> ret;
    #ifdef ARRAY_DEBUG
      ret.setup_arr( myname.c_str() , rank , dimension );
    #else
      ret.setup_arr( ""             , rank , dimension );
    #endif
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems*sizeof(T),cudaMemcpyHostToDevice,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems*sizeof(T),hipMemcpyHostToDevice,0);
        hipDeviceSynchronize();
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems*sizeof(T),cudaMemcpyDeviceToDevice,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems*sizeof(T),hipMemcpyDeviceToDevice,0);
        hipDeviceSynchronize();
      #endif
    }
    return ret;
  }


  inline void deep_copy(Array<T,memHost> lhs) {
    if (myMem == memHost) {
      for (int i=0; i<totElems; i++) {
        lhs.myData[i] = myData[i];
      }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems*sizeof(T),cudaMemcpyDeviceToHost,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems*sizeof(T),hipMemcpyDeviceToHost,0);
      #endif
    }
  }


  inline void deep_copy(Array<T,memDevice> lhs) {
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems*sizeof(T),cudaMemcpyHostToDevice,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems*sizeof(T),hipMemcpyHostToDevice,0);
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems*sizeof(T),cudaMemcpyDeviceToDevice,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems*sizeof(T),hipMemcpyDeviceToDevice,0);
      #endif
    }
  }


  /* ACCESSORS */
  YAKL_INLINE int get_rank() const {
    return rank;
  }
  YAKL_INLINE size_t get_totElems() const {
    return totElems;
  }
  YAKL_INLINE size_t const *get_dimensions() const {
    return dimension;
  }
  YAKL_INLINE T *data() const {
    return myData;
  }
  YAKL_INLINE T *get_data() const {
    return myData;
  }
  YAKL_INLINE size_t extent( int const dim ) const {
    return dimension[dim];
  }
  YAKL_INLINE int extent_int( int const dim ) const {
    return (int) dimension[dim];
  }

  YAKL_INLINE int span_is_contiguous() const {
    return 1;
  }
  YAKL_INLINE int use_count() const {
    return *refCount;
  }
  #ifdef ARRAY_DEBUG
    const char* label() const {
      return myname.c_str();
    }
  #endif


  /* INFORM */
  inline void print_rank() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    std::cout << "Number of Dimensions: " << rank << "\n";
  }
  inline void print_totElems() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    std::cout << "Total Number of Elements: " << totElems << "\n";
  }
  inline void print_dimensions() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    std::cout << "Dimension Sizes: ";
    for (int i=0; i<rank; i++) {
      std::cout << dimension[i] << ", ";
    }
    std::cout << "\n";
  }
  inline void print_data() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    if (rank == 1) {
      for (size_t i=0; i<dimension[0]; i++) {
        std::cout << std::setw(12) << (*this)(i) << "\n";
      }
    } else if (rank == 2) {
      for (size_t j=0; j<dimension[0]; j++) {
        for (size_t i=0; i<dimension[1]; i++) {
          std::cout << std::setw(12) << (*this)(i,j) << " ";
        }
        std::cout << "\n";
      }
    } else if (rank == 0) {
      std::cout << "Empty Array\n\n";
    } else {
      for (size_t i=0; i<totElems; i++) {
        std::cout << std::setw(12) << myData[i] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }


  /* OPERATOR<<
  Print the array. If it's 2-D, print a pretty looking matrix */
  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
    #ifdef ARRAY_DEBUG
      os << "For Array labeled: " << v.myname << "\n";
    #endif
    os << "Number of Dimensions: " << v.rank << "\n";
    os << "Total Number of Elements: " << v.totElems << "\n";
    os << "Dimension Sizes: ";
    for (int i=0; i<v.rank; i++) {
      os << v.dimension[i] << ", ";
    }
    os << "\n";
    if (v.rank == 1) {
      for (size_t i=0; i<v.dimension[0]; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (v.rank == 2) {
      for (size_t j=0; j<v.dimension[1]; j++) {
        for (size_t i=0; i<v.dimension[0]; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else if (v.rank == 0) {
      os << "Empty Array\n\n";
    } else {
      for (size_t i=0; i<v.totElems; i++) {
        os << v.myData[i] << " ";
      }
      os << "\n";
    }
    os << "\n";
    return os;
  }


};

}

#endif
