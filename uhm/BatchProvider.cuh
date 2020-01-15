/** @file BatchProvider.cuh */
#include "Batch.cuh"
#include "GPUHMBase.cuh"
#include "hash.cuh"
#include <iostream>
#include <mutex>

#ifndef BATCHPROVIDER_CUH
#define BATCHPROVIDER_CUH
namespace uhm {
/**
 * Thread safe batch provider
 */
template <typename Key, typename Value> //
class BatchProvider {
public:
  BatchProvider() = delete;
  /**
   * Create batch provider
   * @param num_gpus number of GPUs
   */
  explicit BatchProvider(int num_gpus) : num_gpus_(num_gpus) {
    // Allocate space for a pointer to the first batch, per GPU.
    list_heads_ = new Batch<Key, Value> *[num_gpus_];
    for (int i = 0; i < num_gpus_; ++i) {
      list_heads_[i] = nullptr;
    }
    // Allocate pointers to point to the next batch to be processed.
    next_batches_ = new Batch<Key, Value> *[num_gpus_];
    for (int i = 0; i < num_gpus_; ++i) {
      next_batches_[i] = nullptr;
    }
    // Allocate pointers to point to the tail of the list of batches.
    list_tails_ = new Batch<Key, Value> *[num_gpus_];
    for (int i = 0; i < num_gpus_; ++i) {
      list_tails_[i] = nullptr;
    }
  }

  /**
   * Adds a new batch to the list of batches for GPU gpu_idx. It assumes that no
   * batches have been dispatched.
   */
  void AddBatch(int gpu_idx, Batch<Key, Value> *new_batch) {
    // coarse lock
    mtx.lock();
    // std::clog << "[TRACE] Batch being added\n";
    // When a new batch is added, append it to the tail of batches for the given
    // GPU then update the tail pointer to point to the newly added batch.
    if (list_tails_[gpu_idx] != nullptr) {
      // std::clog << "[TRACE] Batch list not empty for GPU " << gpu_idx <<
      // "\n";
      list_tails_[gpu_idx]->SetNext(new_batch);
    } else {
      // If we are adding the first batch we must set the head of the lists and
      // init the next pointer as well.
      // std::clog << "[TRACE] Batch list empty for GPU " << gpu_idx << "\n";
      list_heads_[gpu_idx] = new_batch;
      next_batches_[gpu_idx] = new_batch;
    }
    list_tails_[gpu_idx] = new_batch;
    if (next_batches_[gpu_idx] == nullptr) {
      next_batches_[gpu_idx] = new_batch;
    }
    // unlock
    mtx.unlock();
  }

  /**
   * Returns true if there are no more batches for any of the GPUs. In other
   * words, the next batch pointers are all nullptr.
   */
  bool IsEmpty() {
    // can make read lock
    mtx.lock();
    for (int gpu_idx = 0; gpu_idx < num_gpus_; ++gpu_idx) {
      if (next_batches_[gpu_idx] != nullptr) {
        // std::clog << "[TRACE] Provider is not empty\n";
        mtx.unlock();
        return false;
      }
    }
    // std::clog << "[TRACE] Provider is empty\n";
    mtx.unlock();
    return true;
  }

  /**
   * Checks if the provider has the next batch
   */
  bool HasNextBatch(int gpu_idx) {
    mtx.lock();
    bool b = (next_batches_[gpu_idx] != nullptr);
    mtx.unlock();
    return b;
  }

  /**
   * Returns the next batch to process given the GPU. This should only be called
   * after ensuring that there is a batch to process for the GPU.
   */
  Batch<Key, Value> *GetNextBatchForGPU(int gpu_idx) {
    mtx.lock();
    if (next_batches_[gpu_idx] == nullptr) {
      // std::clog << "[Batch::GetNextBatch] No more batches to process for GPU"
      // << gpu_idx << std::endl;
      mtx.unlock();
      return nullptr;
    } else {
      // Grab the next batch to be processed and move the pointer to the next
      // batch.
      // std::clog << "[Batch::GetNextBatch] Providing next batch for GPU" <<
      // gpu_idx << std::endl;

      Batch<Key, Value> *curr_batch = next_batches_[gpu_idx];
      next_batches_[gpu_idx] = curr_batch->Next();
      mtx.unlock();
      return curr_batch;
    }
  }

  /** Resets all batches in the batch provider. */
  void ResetAllBatches() {
    mtx.lock();
    for (int gpu_idx = 0; gpu_idx < numgpu; ++gpu_idx) {
      Batch<Key, Value> *curr_batch = list_heads_[gpu_idx];
      Batch<Key, Value> *next_batch;
      while (curr_batch != nullptr) {
        next_batch = curr_batch->Next();
        curr_batch->ResetRequestTypes();
        curr_batch->ResetResponseResults();
        curr_batch = next_batch;
      }
    }
    ResetNextPointers();
    CalculateAllBatchSizes();
    mtx.unlock();
  }

private:
  /**
   *Calculates and sets the size of each batch in the batch provider. Must be
   *called after overwritting existing batches.
   * @warning not thread safe lock must be held
   */
  void CalculateAllBatchSizes() {
    for (int gpu_idx = 0; gpu_idx < num_gpus_; ++gpu_idx) {
      Batch<Key, Value> *curr_batch = list_heads_[gpu_idx];
      Batch<Key, Value> *next_batch;
      while (curr_batch != nullptr) {
        next_batch = curr_batch->Next();
        curr_batch->CalculateSize();
        curr_batch = next_batch;
      }
    }
  }

  /**
   *Resets the next pointers.
   * @warning not thread safe lock must be held
   */
  void ResetNextPointers() {
    for (int gpu_idx = 0; gpu_idx < num_gpus_; ++gpu_idx) {
      next_batches_[gpu_idx] = list_heads_[gpu_idx];
    }
  }

  /**
   *Returns a pointer to the next batch.
   * @warning This is designed only to allow overwritting existing batches.
   */
  Batch<Key, Value> *GetHead(int gpu_idx) { return list_heads_[gpu_idx]; }

  // The number of GPUs to provide batches for.
  int num_gpus_;
  // A dynamically allocated array of pointers to the head of the batch list.
  // Each GPU gets its own list of batches to work from.
  Batch<Key, Value> **list_heads_;
  // A dynamically allocated array of pointers to the tail of each list. This is
  // where new batches will be added.
  Batch<Key, Value> **list_tails_;
  // An array of pointers to the next batch to be processes.
  Batch<Key, Value> **next_batches_;
  // A mutex
  std::mutex mtx;
};
} // namespace uhm
#endif