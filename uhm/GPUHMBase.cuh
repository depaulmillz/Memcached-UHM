/** @file GPUHMBase.cuh */
#ifndef GPUHMBASE_CUH
#define GPUHMBASE_CUH

#define REQUEST_GET 1
#define REQUEST_SET 2
#define REQUEST_INSERT 3
#define REQUEST_REMOVE 4

namespace uhm {
/**
 * Number of GPU
 */
int numgpu = 1;
/**
 * Number of blocks
 */
unsigned num_blocks = 6;
/**
 * Number of threads per block
 */
unsigned threads_per_block = 512;
/**
 * Requests Per Thread
 */
unsigned requests_per_thread = 4;
/**
 * Batch size parameter
 */
unsigned batch_size = num_blocks * threads_per_block * requests_per_thread;

/**
 * Tuple Data Type for the GPU
 */
template <class K, class V> //
struct Tuple {
  int isvalid = 1;
  K key;
  V value;
};

/**
 * Vector data type for the GPU
 */
template <class K, class V> //
struct Vector {
  Tuple<K, V> *elements;
  unsigned capacity;
  unsigned size;
  int wlock = 0;   // 8 bytes
  int rlock = 0;   // 4 bytes
  int readers = 0; // 4 bytes
};

/**
 * GPUTable is a datatype to store a single GPUTable of buckets
 */
template <class K, class V> //
struct GPUTable {
  long size;
  Vector<K, V> *buckets;
};
} // namespace uhm

#endif