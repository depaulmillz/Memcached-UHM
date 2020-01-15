/** @file Batch.cuh */
#include "Allocator.cuh"
#include "GPUHMBase.cuh"
#include <assert.h>
#include <atomic>
#ifndef BATCH_CUH
#define BATCH_CUH
namespace uhm {
/**
 *Batch is used to store requests and responses. It wraps a list
 *of Tuple objects to make it easier to dispatch batches to GPUs.
 */
template <typename Key, typename Value> //
class Batch {
public:
  Batch() = delete;
  /** Creates batch in unified memory.
   *  Batches are uninitialized when they are created. It is the responsibility
   * of the creator to fill them. A group id must be passed to indicate where to
   * allocate the memory.
   */
  explicit Batch(int size, int group_id) : capacity_(size), group_id_(group_id) {
    umallocate(&request_values_, sizeof(Tuple<Key, Value>) * capacity_, group_id_);
    umallocate(&request_types_, sizeof(int *) * capacity_, group_id_);
    umallocate(&response_values_, sizeof(Tuple<Key, Value>) * capacity_, group_id_);
    umallocate(&response_results_, sizeof(int *) * capacity_, group_id_);
    next_ = nullptr;
    size_ = 0;
    processedSignal = false;
  }

  // Setters
  // Used after Batch creation to set records.
  /**
   * Set the Request Value at index to request
   * @param index
   * @param request
   */
  void SetRequestValue(int index, const Tuple<Key, Value> request) {
    // Copy the new record into the desired index.
    assert(index >= 0 && index < capacity_);
    request_values_[index].key = request.key;
    request_values_[index].value = request.value;
    request_values_[index].isvalid = request.isvalid;
  }
  /**
   * Set the Request type at index to type
   * @param index
   * @param type
   */
  void SetRequestType(int index, int type) {
    assert(index >= 0 && index < capacity_);
    request_types_[index] = type;
  }

  /**
   * Set the response value at index to response
   * @param index
   * @param response
   */
  void SetResponseValue(int index, const Tuple<Key, Value> response) {
    assert(index >= 0 && index < capacity_);
    response_values_[index].key = response.key;
    response_values_[index].value = response.value;
    response_values_[index].isvalid = response.isvalid;
  }

  /**
   * Set the response result at index to response
   * @param index
   * @param result
   */
  void SetResponseResult(int index, int result) {
    assert(index >= 0 && index < capacity_);
    response_results_[index] = result;
  }

  // Getters

  /**
   *Returns request values
   * @return
   */
  Tuple<Key, Value> *GetRequestValues() const { return request_values_; }

  /**
   *Returns request types
   * @return
   */
  int *GetRequestTypes() const { return request_types_; }

  /**
   *Returns request response values
   * @return
   */
  Tuple<Key, Value> *GetResponseValues() const { return response_values_; }

  /**
   *Returns request response results
   * @return
   */
  int *GetResponseResults() const { return response_results_; }

  /**
   * Clears the request types in this batch.
   */
  void ResetRequestTypes() {
    for (int i = 0; i < capacity_; ++i) {
      request_types_[i] = 0; // EMPTY
    }
  }
  /**
   *Clears the response types in this batch.
   */
  void ResetResponseResults() {
    for (int i = 0; i < capacity_; ++i) {
      response_results_[i] = 0; // EMPTY
    }
  }

  /**
   * Check if batch is empty
   * @return
   */
  bool IsEmpty() {
    for (int i = 0; i < capacity_; ++i) {
      if (request_types_[i] != 0) { // EMPTY
        return false;
      }
    }
    return true;
  }

  /**
   *Sorts a batch so that requests with similar hash values are close
   * together. This is intended to help the GPU process requests quickly.
   */
  void SortRequests(GPUTable<Key, Value> *table) {
    // Insertion sort.
    for (int i = 0; i < capacity_ - 1; ++i) {
      int min_idx = i;
      int curr_hash = hashfunction(&(request_values_[i].key), table->size);
      for (int j = i + 1; j < capacity_; ++j) {
        int temp_hash = hashfunction(&(request_values_[j].key), table->size);
        if (temp_hash < curr_hash) {
          min_idx = j;
        }
      }
      auto temp_value = request_values_[i];
      request_values_[i] = request_values_[min_idx];
      request_values_[min_idx] = temp_value;

      auto temp_type = request_types_[i];
      request_types_[i] = request_types_[min_idx];
      request_types_[min_idx] = temp_type;
    }
  }

  /**
   * Calculate the size of the batch
   * @return size
   */
  int CalculateSize() {
    // Loop through and count the number of non-EMPTY requests.
    size_ = 0;
    for (int i = 0; i < capacity_; ++i) {
      if (request_types_[i] != 0) { // EMPTY
        ++size_;
      }
    }
    return size_;
  }

  /**
   *Return the next batch in the list.
   */
  Batch<Key, Value> *Next() { return next_; }

  /**
   *Returns true if the batch points to another batch.
   */
  bool HasNext() { return next_ != nullptr; }

  /** Set the next batch. */
  void SetNext(Batch<Key, Value> *batch) { next_ = batch; }

  /** Returns the size of the batch. */
  int GetCapacity() { return capacity_; }

  /**
   * Calculates the size of the current batch by finding the number of
   * non-EMPTY requests.
   * @return
   */
  int GetSize() { return size_; }

  /**
   * Checks signal
   * @return signal value
   */
  bool checkSignal() { return processedSignal.load(); }
  /**
   * Set signal using b
   * @param b
   */
  void setSignal(bool b) { processedSignal = b; }

private:
  int capacity_;
  int size_;
  int group_id_;
  Tuple<Key, Value> *request_values_;
  int *request_types_;
  Tuple<Key, Value> *response_values_;
  int *response_results_;
  Batch<Key, Value> *next_;
  std::atomic<bool> processedSignal;
};

} // namespace uhm
#endif