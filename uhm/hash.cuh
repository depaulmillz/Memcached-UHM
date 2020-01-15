#include "GPUHMBase.cuh"
#include "basetypes.cuh"
/** @file hash.cuh */
#ifndef hash_cuh
#define hash_cuh

namespace uhm {

//--------------------------------------------------------------------------------------------------
// HASH FUNCTIONS
//--------------------------------------------------------------------------------------------------

/**
 * Template hash function for GPU to call to hash data
 */
template <class K> //
__host__ __device__ unsigned hashfunction(K *key, long size);

/**
 * int key, value hash function
 */
template <> //
__host__ __device__ unsigned hashfunction<int>(int *key, long size) {
  // volatile int val = t->size;
  // val *= 2;
  return *key % size;
}

/**
 * custr,int key, value hash function using djb2
 */
template <> //
__host__ __device__ unsigned hashfunction<custr>(custr *key, long size) {
  // djb2 hash
  unsigned long hash = 5381;
  char *str = key->str;
  int c;
  while (c = *str++) {
    hash = ((hash << 5) + hash) + c;
  }
  return hash % size;
}

//--------------------------------------------------------------------------------------------------
} // namespace uhm

#endif