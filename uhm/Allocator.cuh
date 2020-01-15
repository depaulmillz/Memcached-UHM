/** @file Allocator.cuh
 * Allocation and deallocation functionality resides here.
 */
#ifndef ALLOCATOR_CUH
#define ALLOCATOR_CUH

#include "gpuErrchk.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace uhm {

const unsigned long long chunkSize = 1024 * 1024 * 1024; // 1 GiB
unsigned long long total_bytes = 0;
int num_4_gb_allocs = 0;
int total_allocs = 0;
int num_big_allocs = 0;

/**
 * Gets padding of a type
 */
size_t getPadding(size_t startingptr, size_t alignment) {
  size_t multiplier = startingptr / alignment + 1;
  size_t padding = multiplier * alignment - startingptr;
  return padding;
}

/**
 * Page size to allocate
 */
const size_t page_size = 1UL << 30; // 1 GiB

/**
 * Allocates memory in a page set by page_size.
 */
class InPageAllocator {
public:
  /**
   * Create the in page allocator.
   */
  InPageAllocator() {
    gpuErrchk(cudaMallocManaged(&this->mem, page_size));
    ++num_4_gb_allocs;
    pos = mem;
  }
  /**
   * Deletes in page allocator, cuda context must exist to do so.
   */
  ~InPageAllocator() { gpuErrchk(cudaFree(this->mem)); }

  /**
   * Allocates memory of type T and sets *ptr to this memory of size s
   * @param ptr
   * @param s
   */
  template <class T> //
  void allocate(T **ptr, size_t s) {
    size_t padding = getPadding((size_t)pos, std::alignment_of<T *>());
    m.lock();
    if (pos + padding + s > mem + page_size) {
      // std::clog << "[DEBUG] pos = " << (void *)pos << ", mem = " << (void
      // *)mem << "\n";
      *ptr = NULL;
    } else {
      *ptr = (T *)(pos + padding);
      pos = (char *)(pos + padding + s);
    }
    total_bytes += (padding + s);
    m.unlock();
  }

private:
  char *pos;
  char *mem;
  std::mutex m;
};

/**
 * Allocates multiple pages of memory
 */
class MultiPageAllocator {
public:
  /**
   * Constructor
   */
  MultiPageAllocator() {}

  /**
   * Delete function
   */
  ~MultiPageAllocator() {
    m.lock();
    for (auto &e : mem) {
      gpuErrchk(cudaFree(e));
    }
    m.unlock();
  }

  /**
   * Allocates memory of type T and sets *ptr to this memory of size s
   * @param ptr
   * @param s
   */
  template <class T> //
  void allocate(T **ptr, size_t s) {
    size_t pages_needed = (size_t)ceil(s / (double)page_size);
    char *c;
    gpuErrchk(cudaMallocManaged(&c, pages_needed * page_size));
    std::clog << "[DEBUG] MultiPageAllocator allocated "
              << pages_needed * page_size / 1024.0 / 1024.0 / 1024.0 << " GB\n";

    *ptr = (T *)c;
    ++num_big_allocs;
    total_bytes += pages_needed * page_size;
    m.lock();
    mem.push_back(c);
    m.unlock();
  }

private:
  std::list<char *> mem;
  std::mutex m;
};

/**
 * Allocates with group affinity
 */
class GroupAllocator {
public:
  /**
   * Constructor takes group_num to allocate to
   * @param group_num
   */
  GroupAllocator(int group_num) : group_num_(group_num) {
    mpa = new MultiPageAllocator();
  }

  /**
   * Delete function
   */
  ~GroupAllocator() {}
  /**
   * Function to free all memory of group allocator
   */
  void freeall() {
    for (auto &e : ipas) {
      delete e;
    }
    delete mpa;
  }

  /**
   * Allocates memory of type T and sets *ptr to this memory of size s
   * @param ptr
   * @param s
   */
  template <class T> //
  void allocate(T **ptr, size_t s) {
    if (ptr == NULL || s == 0) {
      return;
    }

    if (s > page_size) {
      mpa->allocate<T>(ptr, s);
    } else {
      m.lock();
      int lastSize = ipas.size();
      if (lastSize == 0) {
        InPageAllocator *ipa_new = new InPageAllocator();
        printf("[TRACE] [group %d] New %luGB chunk allocated.\n", group_num_,
               page_size >> 30);
        ipas.push_back(ipa_new);
      }
      auto ipa = ipas[ipas.size() - 1];
      m.unlock();
      ipa->allocate<T>(ptr, s);
      while (*ptr == NULL) {
        InPageAllocator *ipa2 = new InPageAllocator();
        m.lock();
        if (lastSize == ipas.size()) {
          ipas.push_back(ipa2);
          lastSize = ipas.size();
        }
        m.unlock();
        m.lock();
        auto ipa = ipas[ipas.size() - 1];
        m.unlock();
        ipa->allocate<T>(ptr, s);
      }
    }
  }

private:
  std::vector<InPageAllocator *> ipas;
  MultiPageAllocator *mpa;
  std::mutex m;
  int group_num_;
};

static std::mutex groupMapMutex;
static std::unordered_map<int, std::shared_ptr<GroupAllocator>> allocator;

/**
 * Allocates memory of type T and sets *ptr to this memory of size s. It
 * allocates in group group.
 * Thread Safe!
 * @param ptr
 * @param s
 * @param group
 */
template <typename T> //
cudaError_t umallocate(T **ptr, size_t s, int group = -1) {
  groupMapMutex.lock();
  std::shared_ptr<GroupAllocator> g = allocator[group];
  if (g == nullptr) {
    g = std::make_shared<GroupAllocator>(group);
    allocator[group] = g;
  }
  groupMapMutex.unlock();

  g->allocate<T>(ptr, s);
  ++total_allocs;

  return cudaSuccess;
}

/**
 * Cleans up the allocator by freeing everything so there is no memory leak.
 * Thread safe.
 */
void cleanUpAllocator() {
  groupMapMutex.lock();
  for (std::pair<const int, std::shared_ptr<uhm::GroupAllocator>> &elm :
       allocator) {
    elm.second->freeall();
  }
  groupMapMutex.unlock();
}

/**
 * Prints allocator statistics
 */
void print_allocator_stats() {
  printf("Total number of allocations made = %d\n\t1GB allocations = "
         "%d\n\tnon-1GB allocations = %d\n\tbig allocations = %d\nTotal bytes "
         "allocated = %llu\n",
         total_allocs, num_4_gb_allocs, total_allocs - num_4_gb_allocs,
         num_big_allocs, total_bytes);
}

} // namespace uhm
#endif