/** @file GPUKVServe.cuh */
#ifndef GPUKVSERVE_CUH
#define GPUKVSERVE_CUH

#include "Allocator.cuh"
#include "BatchProvider.cuh"
#include "GPUHMBase.cuh"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cooperative_groups.h>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#define EMPTY 0
#define DONE -1
#define PROCESS -2
#define READY -3

#define CPU_GROUP -1
#define VECTOR_CAP 10

#define BILLION 1000000000

namespace uhm {

// This type represents a tuple used for timing. It consists of a duration which
// is the length of execution, an unsigned long for the number of operations
// performed and an integer as an identifier if needed.
using time_tuple_t = std::tuple<std::chrono::nanoseconds, unsigned long, int>;

/**
 * This function is the serving function for the GPU
 */
template <class Key, class Value> //
__global__ void uhto_batch_serve(
    const int gpu_idx,
    volatile Tuple<Key, Value> **curr_batch_request_values_ptr,
    volatile Tuple<Key, Value> **curr_batch_response_values_ptr,
    volatile int **curr_batch_request_types_ptr,
    volatile int **curr_batch_response_results_ptr, GPUTable<Key, Value> *t,
    volatile int *terminate_signal, const int requests_per_thread,
    volatile long long *channel_to_cpu, volatile long long *channel_to_gpu) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  volatile Tuple<Key, Value> *curr_request_values = nullptr;
  volatile Tuple<Key, Value> *curr_response_values;
  volatile int *curr_request_types;
  volatile int *curr_response_results;
  volatile long long finished_request_values = 0;

  // Device wide cooperative thread group.
  auto grid = cooperative_groups::this_grid();

  grid.sync();
  int round = 0;
  while (!*terminate_signal) {
    if (tid == 0) {
      while (*channel_to_gpu == (long long)curr_request_values &&
             !*terminate_signal)
        ;
      // printf("[GPU%d]: PROCESS received from CPU (round %d)\n", gpu_idx,
      // round);
    }
    grid.sync();
    if (*terminate_signal) {
      return;
    }
    while ((long long)*(curr_batch_request_values_ptr) ==
           finished_request_values)
      ;
    /*{
      if (tid == 0) {
        printf("Current batch request values ptr %p %p\n",
    *(curr_batch_request_values_ptr), (void *)finished_request_values);
      }
    }*/

    finished_request_values = (long long)curr_request_values;
    curr_request_values = *(curr_batch_request_values_ptr);
    curr_response_values = *(curr_batch_response_values_ptr);
    curr_request_types = *(curr_batch_request_types_ptr);
    curr_response_results = *(curr_batch_response_results_ptr);

    // Synchronize all threads on the device to ensure that values are known by
    // each thread.
    grid.sync();

    // Print information.
    // if (tid == 0) {
    //  printf("[GPU%d]\n", gpuIdx);
    // printf("\tcurr_batch_request_values_ptr = %p\n",
    // curr_batch_request_values_ptr);
    //   printf("\trequestBuffer = %p\n", requestBuffer);
    // printf("\tcurr_batch_response_values_ptr = %p\n",
    // curr_batch_response_values_ptr);
    //  printf("\tcurr_response_values = % p\n ", curr_response_values);
    //  printf("\tbufferInfo_ptr = % p\n ", bufferInfo_ptr);
    // printf("\tinfoToCPU_ptr = %p\n", infoToCPU_ptr);
    // printf("\tt = %p\n", t);
    // printf("\tgpuIdx = %d\n", gpuIdx);
    // printf("\tsignaldone = %p\n", signaldone);
    // printf("\trequests_per_thread = %d\n", requests_per_thread);
    // printf("\tptr_com_to_gpu = %p\n", ptr_com_to_gpu);
    // printf("\tptr_com_to_cpu = %p\n", ptr_com_to_cpu);
    // printf("\tblockInt = %p\n", blockInt);
    // printf("\tgpuDim = %d\n", gpuDim);
    //}

    if (tid == 0) {
      *channel_to_cpu = (long long)curr_request_values;
      __threadfence_system();
    }
    grid.sync();

    // If there are requests to process, do so.
    if (curr_request_values != NULL && curr_response_values != NULL &&
        curr_request_types != NULL && curr_response_results != NULL) {
      for (int req_i = 0; req_i < requests_per_thread; req_i++) {
        int operating_on = requests_per_thread * tid + req_i;

        unsigned long pos = 0;
        volatile auto l = t->buckets;

        if (curr_request_types[operating_on] != EMPTY) {
          pos = hashfunction((Key *)&(curr_request_values[operating_on].key),
                             t->size);
          l += pos;
        }

        volatile int done = 0;

        if (curr_request_types[operating_on] == REQUEST_GET) {
          const Key searching_for = curr_request_values[operating_on].key;

          while (!done) {
            if (atomicCAS_system((int *)&(l->rlock), 0, tid + 1) == 0) {
              l->readers++;
              if (l->readers == 1) {
                volatile int done2 = 0;
                while (!done2) {
                  if (atomicCAS_system((int *)&(l->wlock), 0, tid + 1) == 0) {
                    done2 = 1;
                  }
                }
              }
              done = 1;
              __threadfence_system();
              atomicExch_system((int *)&(l->rlock), 0);
            }
          }
          curr_response_values[operating_on].key = searching_for;
          int validity = 0;
          Value value_at_key;

          for (int i = 0; i < l->size && l->elements[i].key <= searching_for;
               i++) {
            if (l->elements[i].key == searching_for) {
              validity = 1;
              value_at_key = l->elements[i].value;
              break;
            }
          }

          curr_response_values[operating_on].isvalid = validity;
          Value *nonVolatilePointer =
              (Value *)&curr_response_values[operating_on].value;
          *nonVolatilePointer = (Value)value_at_key;

          done = 0;
          while (!done) {
            if (atomicCAS_system((int *)&(l->rlock), 0, tid + 1) == 0) {
              l->readers--;
              __threadfence_system();
              if (l->readers == 0) {
                atomicExch_system((int *)&(l->wlock), 0);
              }
              done = 1;
              __threadfence_system();
              atomicExch_system((int *)&(l->rlock), 0);
            }
          }
          curr_response_results[operating_on] = DONE;

        } else if (curr_request_types[operating_on] == REQUEST_INSERT) {
          const Key searching_for = curr_request_values[operating_on].key;
          __threadfence_system();
          while (!done) {
            if (atomicCAS_system((int *)&(l->wlock), 0, tid + 1) == 0) {
              if (l->size == l->capacity) {
                curr_response_values[operating_on].isvalid = 0;
                printf("Error it is full\n");
              } else {
                int i;
                for (i = 0; i < l->size; i++) {
                  if (l->elements[i].key >= searching_for) {
                    break;
                  }
                }
                if (l->elements[i].key != searching_for) {
                  for (int j = l->size; j > i; --j) {
                    l->elements[j].key = l->elements[j - 1].key;
                    l->elements[j].value = l->elements[j - 1].value;
                  }
                  l->elements[i].key =
                      (Key)curr_request_values[operating_on].key;
                  l->elements[i].value =
                      (Value)curr_request_values[operating_on].value;

                  curr_response_values[operating_on].isvalid = 1;
                  l->size++;
                }
              }
              __threadfence_system();
              done = 1;
              atomicExch_system((int *)&(l->wlock), 0);
            }
          }
          curr_response_results[operating_on] = DONE;

        } else if (curr_request_types[operating_on] == REQUEST_SET) {
          const Key searching_for = curr_request_values[operating_on].key;

          while (!done) {
            if (atomicCAS_system((int *)&(l->wlock), 0, tid + 1) == 0) {
              int validity = 0;

              for (int i = 0; i < l->size; i++) {
                if (l->elements[i].key == searching_for) {
                  validity = 1;
                  l->elements[i].value =
                      (Value)curr_request_values[operating_on].value;
                  break;
                }
              }

              curr_response_values[operating_on].isvalid = validity;

              done = 1;
              __threadfence_system();
              atomicExch_system((int *)&(l->wlock), 0);
            }
          }
          curr_response_results[operating_on] = DONE;
        } else if (curr_request_types[operating_on] == REQUEST_REMOVE) {
          const Key searching_for = curr_request_values[operating_on].key;

          while (!done) {
            if (atomicCAS_system((int *)&(l->wlock), 0, tid + 1) == 0) {
              int validity = 0;
              int i;

              // Search for the key to remove.
              for (i = 0; i < l->size; i++) {
                if (l->elements[i].key == searching_for) {
                  validity = 1;
                  break;
                }
              }

              // If the key was found, remove it by shifting over all
              // elements in the vector.
              for (; validity == 1 && i < l->size - 1; i++) {
                l->elements[i] = l->elements[i + 1];
              }

              if (validity == 1) {
                (l->size)--;
              }

              curr_response_values[operating_on].isvalid = validity;

              done = 1;
              __threadfence_system();

              atomicExch_system((int *)&(l->wlock), 0);
            }
          }
          curr_response_results[operating_on] = DONE;
        }
      }
    }
    ++round;
    grid.sync();
  }
}

/**
 * UnifiedHashTableOnline is a class wrapping up the functionality needed to
 * serve a hash map from the GPU
 */
template <class Key, class Value> //
class UnifiedHashTableOnline {
public:
  UnifiedHashTableOnline() = delete;
  /**
   * Create a UnifiedHashTableOnline with size buckets and req_per_th requests
   * per thread.
   */
  UnifiedHashTableOnline(int size, int req_per_th)
      : requests_per_thread_(req_per_th), cached_size(size) {
    // Set the number of active threads.
    // TODO getnumgpu();
    num_serving_threads_ = num_blocks * threads_per_block * numgpu;

    // Create the table to store elements. We assume that they will eventually
    // be paged to the desired GPU.
    umallocate(&this->t_, sizeof(GPUTable<Key, Value>));
    t_->size = size;
    umallocate(&this->t_->buckets, sizeof(Vector<Key, Value>) * size);

    // Allocate and initialize each vector in the table.
    for (int i = 0; i < t_->size; i++) {
      umallocate(&this->t_->buckets[i].elements,
                 sizeof(Tuple<Key, Value>) * VECTOR_CAP);
      t_->buckets[i].capacity = VECTOR_CAP;
      t_->buckets[i].size = 0;
    }

    // Allocate and init the pointers to the current batch's response values.
    umallocate(&curr_batch_response_values_ptrs_,
               sizeof(Tuple<Key, Value> **) * numgpu, CPU_GROUP);
    for (int i = 0; i < numgpu; i++) {
      // Allocate a pointer in the GPUs group that will be updated with a
      // pointer to the head of the batch.
      umallocate(&(curr_batch_response_values_ptrs_[i]),
                 sizeof(Tuple<Key, Value> *), i);
      *(curr_batch_response_values_ptrs_[i]) = nullptr;
    }

    // Allocate and init pointers to the response results.
    umallocate(&curr_batch_response_results_ptrs_,
               sizeof(volatile int **) * numgpu, CPU_GROUP);
    for (int i = 0; i < numgpu; i++) {
      // Same as previous but with the results.
      umallocate(&(curr_batch_response_results_ptrs_[i]),
                 sizeof(volatile int *), i);
      *(curr_batch_response_results_ptrs_[i]) = nullptr;
    }

    // Set up the communication channel to the CPU.
    umallocate(&channels_to_cpu_, sizeof(volatile long long *) * numgpu,
               CPU_GROUP);
    for (int i = 0; i < numgpu; i++) {
      umallocate(&(channels_to_cpu_[i]), sizeof(volatile long long), i);
      *(channels_to_cpu_[i]) = 0ll;
    }

    // Set up request pointers.
    umallocate(&curr_batch_request_values_ptrs_,
               sizeof(Tuple<Key, Value> **) * numgpu, CPU_GROUP);
    for (int i = 0; i < numgpu; ++i) {
      umallocate(&(curr_batch_request_values_ptrs_[i]),
                 sizeof(Tuple<Key, Value> *), i);
      *(curr_batch_request_values_ptrs_[i]) = nullptr;
    }

    // Set up request type pointers.
    umallocate(&curr_batch_request_types_ptrs_,
               sizeof(volatile int **) * numgpu, CPU_GROUP);
    for (int i = 0; i < numgpu; ++i) {
      umallocate(&(curr_batch_request_types_ptrs_[i]), sizeof(volatile int *),
                 i);
      *(curr_batch_request_types_ptrs_[i]) = EMPTY;
    }

    umallocate(&channels_to_gpu_, sizeof(volatile long long *) * numgpu,
               CPU_GROUP);
    for (int i = 0; i < numgpu; ++i) {
      umallocate(&(channels_to_gpu_[i]), sizeof(volatile long long), i);
      *(channels_to_gpu_[i]) = 0ll;
    }

    // Signal to indicate kernel may terminate. Is an array of integer pointers
    // that are passed during kernel launch. GPU will dereference pointer to
    // check if it should terminate.
    umallocate(&terminate_signal_ptrs_, sizeof(volatile int *) * numgpu);
    for (int i = 0; i < numgpu; ++i) {
      umallocate(&terminate_signal_ptrs_[i], sizeof(volatile int), i);
      *(terminate_signal_ptrs_[i]) = 0;
    }
  }

  /**
   * Deletes the object
   */
  ~UnifiedHashTableOnline() {}

  /**
   * Creates a buffer
   */
  static std::tuple<Tuple<Key, Value> **, int *> createBuffer(size_t size,
                                                              int groupId) {
    // Allocate the buffer and communication channel.
    Tuple<Key, Value> **data_1;
    umallocate(&data_1, sizeof(Tuple<Key, Value> *) * size, groupId);
    int *data_2;
    umallocate(&data_2, sizeof(int) * size, groupId);

    // Initialize the values of the buffer and communication channel.
    for (int i = 0; i < size; i++) {
      data_1[i] = NULL;
      data_2[i] = EMPTY;
    }

    return std::make_tuple(data_1, data_2);
  }

  /**
   * Runs batches of read and write operations
   */
  void readWriteBatch(BatchProvider<Key, Value> &batch_provider) {
    // std::clog << "[TRACE] Batching started on GPU\n";
    // Set up timing vector.
    std::vector<time_tuple_t> run_times;

    // Reset termination signals.
    for (int i = 0; i < numgpu; ++i) {
      *(terminate_signal_ptrs_[i]) = 0;
      *(channels_to_gpu_[i]) = 0ll;
      *(channels_to_cpu_[i]) = 0ll;
    }

    // Launch GPU exectuion.
    this->serve();
    Batch<Key, Value> **lastBatch =
        single_dispatcher(batch_provider, &run_times);

    for (int i = 0; i < numgpu; ++i) {
      __sync_synchronize();
      *(terminate_signal_ptrs_[i]) = 1;
      // printf("Terminate signal sent to GPU%d\n", gpu_idx);
      __sync_synchronize();
    }
    /*
        double average_throughput = 0.0;
        std::chrono::nanoseconds total_duration(0);
        unsigned long total_num_ops = 0;
        for (const auto &it : run_times) {
          // Accumulate the durations and number of operations from each round,
       then
          // print out the round throughput.

          const auto curr_duration = std::get<0>(it);
          const auto curr_num_ops = std::get<1>(it);
          const int curr_round = std::get<2>(it);
          const double curr_throughput = (1000 * (double)curr_num_ops) /
       (double)curr_duration.count(); total_duration += curr_duration;
          total_num_ops += curr_num_ops;
          average_throughput += curr_throughput;
          // A duration is in nano seconds and the below calculation outputs
       Mops/s.
          // It is derived from (#ops / (duration / BILLION)) / MILLION.
          printf("[ROUND %d Throughput] %4.3f Mops/s (ops = %lu, duration = %lu
       ns)\n", curr_round, curr_throughput, curr_num_ops,
       curr_duration.count());

          std::clog << "[INFO] round, Mops/s, ops, duration (ns)\n"
                    << curr_round << ", " << std::setprecision(3) <<
       curr_throughput << ", " << curr_num_ops << ", "
                    << curr_duration.count() << "\n";
        }
        average_throughput /= run_times.size();
        printf("\tTotal throughput = %4.3f Mops/s (ops = %lu, duration = %lu
       ns)\n", (1000 * (double)total_num_ops) / (double)total_duration.count(),
       total_num_ops, total_duration.count()); printf("\tAverage throughput =
       %4.3f Mops/s\n", average_throughput);
    */
    this->finish();

    // signal last batches
    for (int i = 0; i < numgpu; i++) {
      if (lastBatch[i] != nullptr) {
        // std::clog << "[TRACE] Just signaled batch\n";
        lastBatch[i]->setSignal(true);
      }
    }

    delete[] lastBatch;

    // std::clog << "[TRACE] Batching ended on GPU\n";
  }

  /** Return the hash value of the given key for this table. */
  int HashFunction(int key) { return hashfunction(&key, t_->size); }

  /**
   * Returns the table for this UHM
   */
  GPUTable<Key, Value> *GetTable() { return t_; }

  /// Returns the size of the hash table
  unsigned GetSize() { return cached_size; }

private:
  // An array of pointers (one per GPU) to the pointer that will contain the
  // head of the current batch of request to be executed.
  volatile int ***curr_batch_request_types_ptrs_;
  // An array of pointers (one per GPU) to the pointer that will contain the
  // head of the values for the current batch of request to be executed.
  volatile Tuple<Key, Value> ***curr_batch_request_values_ptrs_;
  // An array of pointers (one per GPU) to the pointer that will contain the
  // head of the results for each operation in the current batch.
  volatile int ***curr_batch_response_results_ptrs_;
  // An array of pointers (one per GPU) to the pointer that will contain the
  // head of the results for each operation in the current batch.
  volatile Tuple<Key, Value> ***curr_batch_response_values_ptrs_;
  // An array of channels to the CPU from each GPU.
  volatile long long **channels_to_cpu_;
  // An array of communication channels to each GPU.
  volatile long long **channels_to_gpu_;
  // A pointer per GPU to the flag signalling the that the kernel should
  // terminate.
  volatile int **terminate_signal_ptrs_;

  // Commmon member variables (shared by all GPUs).
  // A pointer to the head of the hashmap where elements are stored.
  GPUTable<Key, Value> *t_;
  // Number of requests to be executed per thread.
  int requests_per_thread_;
  // Total number of threads active on the GPU.
  int num_serving_threads_;

  unsigned cached_size;

  /**
   * Protocol that dispatches from the batch provider
   */
  Batch<Key, Value> **
  single_dispatcher(BatchProvider<Key, Value> &batch_provider,
                    std::vector<time_tuple_t> *run_times) {
    // std::cerr << "Single dispatcher started...\n";

    // Increment through each set of batches to be dispatched.
    int round = 0;
    int batch_num = 0;
    bool gpu_is_active[numgpu];
    bool is_last_batch[numgpu];
    long long lastResponse[numgpu];
    // keeps track of what to signal
    Batch<Key, Value> **lastBatch = new Batch<Key, Value> *[numgpu];
    std::unique_ptr<Batch<Key, Value> *[]> next_batch =
        std::unique_ptr<Batch<Key, Value> *[]>(new Batch<Key, Value> *[numgpu]);

    for (int i = 0; i < numgpu; i++) {
      lastResponse[i] = 0ll;
      lastBatch[i] = nullptr;
      next_batch[i] = nullptr;
    }
    while (!batch_provider.IsEmpty()) {
      // Set up round timing.
      auto start_time = std::chrono::high_resolution_clock::now();
      unsigned long num_ops = 0;

      // Phase 1: Prepare pointers for the GPU and send PROCESS.
      for (int gpu_idx = 0; gpu_idx < numgpu; ++gpu_idx) {
        gpu_is_active[gpu_idx] = false;
        if (batch_provider.HasNextBatch(gpu_idx)) {
          gpu_is_active[gpu_idx] = true;
          // Set the pointer to the head of the batch to execute for this GPU.
          next_batch[gpu_idx] = batch_provider.GetNextBatchForGPU(gpu_idx);
          num_ops += next_batch[gpu_idx]->GetSize();
          if (!next_batch[gpu_idx]->HasNext()) {
            is_last_batch[gpu_idx] = true;
          }
          auto next_request_values =
              (volatile Tuple<Key, Value> *)next_batch[gpu_idx]
                  ->GetRequestValues();
          auto next_response_values =
              (volatile Tuple<Key, Value> *)next_batch[gpu_idx]
                  ->GetResponseValues();
          auto next_request_types =
              (volatile int *)next_batch[gpu_idx]->GetRequestTypes();
          auto next_response_results =
              (volatile int *)next_batch[gpu_idx]->GetResponseResults();
          assert(next_request_values != nullptr);
          assert(next_request_types != nullptr);
          assert(next_response_values != nullptr);
          assert(next_response_results != nullptr);
          *(curr_batch_request_values_ptrs_[gpu_idx]) = next_request_values;
          *(curr_batch_response_values_ptrs_[gpu_idx]) = next_response_values;
          *(curr_batch_request_types_ptrs_[gpu_idx]) = next_request_types;
          *(curr_batch_response_results_ptrs_[gpu_idx]) = next_response_results;

          // Indicate that the GPU may begin processing, ensuring
          // that all previously written values are seen.
          __sync_synchronize();
          *(channels_to_gpu_[gpu_idx]) = (long long)next_request_values;
          __sync_synchronize();
          ++batch_num;
        }
      }

      // Phase 2: Wait for a signal from the GPUs that they have successfully
      // read the pointer values.
      for (int gpu_idx = 0; gpu_idx < numgpu; ++gpu_idx) {
        if (gpu_is_active[gpu_idx]) {
          while (*(channels_to_cpu_[gpu_idx]) == lastResponse[gpu_idx])
            ;
          __sync_synchronize();
          lastResponse[gpu_idx] = *(channels_to_cpu_[gpu_idx]);
          if (lastBatch[gpu_idx] != nullptr) {
            // std::clog << "[TRACE] Just signaled batch\n";
            lastBatch[gpu_idx]->setSignal(true);
          }
          lastBatch[gpu_idx] = next_batch[gpu_idx];
        }
      }

      // Record duration of this round.
      auto stop_time = std::chrono::high_resolution_clock::now();
      std::chrono::nanoseconds duration = (stop_time - start_time);
      run_times->push_back(time_tuple_t(duration, num_ops, round));
      ++round;
    }
    // std::cerr << "CPU dispatcher done...\n";
    return lastBatch;
  }

  void serve() {
    // Struct to hold arguments for cooperative thread kernel launch.
    struct args_t {
      void *gpu_idx;
      void *curr_batch_request_values_ptr;
      void *curr_batch_response_values_ptr;
      void *curr_batch_request_types_ptr;
      void *curr_batch_response_results_ptr;
      void *t;
      void *gpuIdx;
      void *terminate_signal;
      void *requests_per_thread;
      void *channel_to_cpu;
      void *channel_to_gpu;
    };

    volatile int *blockInt;
    umallocate(&blockInt, sizeof(int));
    *blockInt = 0;
    for (int gpu_idx = 0; gpu_idx < numgpu; gpu_idx++) {
      cudaSetDevice(gpu_idx);
      // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
      // uhto_batch_serve<<<num_blocks, threads_per_block>>>(
      //     gpu_idx, curr_batch_request_values_ptrs_[gpu_idx],
      //     curr_batch_response_values_ptrs_[gpu_idx],
      //     curr_batch_request_types_ptrs_[gpu_idx],
      //     curr_batch_response_results_ptrs_[gpu_idx], t_,
      //     terminate_signal_ptrs_[gpu_idx], requests_per_thread_,
      //     channels_to_cpu_[gpu_idx], channels_to_gpu_[gpu_idx]);

      args_t args = {&gpu_idx,
                     &curr_batch_request_values_ptrs_[gpu_idx],
                     &curr_batch_response_values_ptrs_[gpu_idx],
                     &curr_batch_request_types_ptrs_[gpu_idx],
                     &curr_batch_response_results_ptrs_[gpu_idx],
                     &t_,
                     &terminate_signal_ptrs_[gpu_idx],
                     &requests_per_thread_,
                     &channels_to_cpu_[gpu_idx],
                     &channels_to_gpu_[gpu_idx]};

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, gpu_idx);
      gpuErrchk(cudaLaunchCooperativeKernel(
          (void *)uhto_batch_serve<Key, Value>, num_blocks, threads_per_block,
          (void **)&args));
    }
  }

  void finish() {
    for (int gpuIdx = 0; gpuIdx < numgpu; gpuIdx++) {
      cudaSetDevice(gpuIdx);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
    }
  }
};
} // namespace uhm
#endif