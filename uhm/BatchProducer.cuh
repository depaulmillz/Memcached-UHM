/** @file BatchProducer.cuh */

#include "Batch.cuh"
#include "BatchProvider.cuh"
#include "optional.hh"
#include <atomic>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>

#ifndef BATCHPRODUCER_CUH
#define BATCHPRODUCER_CUH
#define TIMEOUT 0.001

namespace uhm { /**
                 * Denotes operation type
                 */
enum OperationType { GET, SET, REMOVE, INSERT };

/**
 * Union of bool and type V
 */
template <typename V> //
union boolOrV {
  bool b;
  V value;
};

/**
 * Produces batches for offloading to batch provider
 */
template <typename Key, typename Value> //
class BatchProducer {
public:
  BatchProducer() {
    Batch<Key, Value> *batch = new Batch<Key, Value>(batch_size, 0);
    for (unsigned i = 0; i < batch_size; i++) {
      batch->SetRequestType((int)i, 0);
    }
    curr_batch.push_back(batch);
    inserted.push_back(0);
    created.push_back(clock());
    done.push_back(false);
  }

  Tuple<Key, Value> request(OperationType t, Key key, optional<Value> value,
                            BatchProvider<Key, Value> &provider) {
    // std::clog << "[TRACE] New request with key \n";
    Batch<Key, Value> *batch;
    int idx = 0;
    mtx.lock();
    int curr_batch_idx = curr_batch.size() - 1;
    if (curr_batch_idx >= 0 && inserted[curr_batch_idx] < batch_size &&
        (double)(clock() - created[curr_batch_idx]) <
            TIMEOUT * CLOCKS_PER_SEC &&
        done[curr_batch_idx] == false) {
      // std::cerr << "[DEBUG] can insert into batch " << curr_batch_idx
      //          << std::endl;
      idx = inserted[curr_batch_idx];
      inserted[curr_batch_idx]++;
      batch = curr_batch[curr_batch_idx];
    } else {
      batch = new Batch<Key, Value>(batch_size, 0);
      for (unsigned i = 0; i < batch_size; i++) {
        batch->SetRequestType((int)i, 0);
      }
      curr_batch.push_back(batch);
      int old_batch_idx = curr_batch_idx;
      curr_batch_idx = curr_batch.size() - 1;
      // std::cerr << "[DEBUG] creating new batch " << curr_batch_idx
      //          << " because done: " << done[old_batch_idx] << " and inserted
      //          "
      //         << inserted[old_batch_idx] << std::endl;

      inserted.push_back(0);
      idx = inserted[curr_batch_idx];
      inserted[curr_batch_idx]++;
      created.push_back(clock());
      done.push_back(false);
    }
    mtx.unlock();
    // std::clog << "[TRACE] request type and value set at idx " << idx << "\n";

    switch (t) {
    case OperationType::GET:
      batch->SetRequestType(idx, REQUEST_GET);
      batch->SetRequestValue(idx, Tuple<Key, Value>{1, key, Value()});
      break;
    case OperationType::SET:
      if (value != optional<Value>(nullopt)) {
        batch->SetRequestType(idx, REQUEST_SET);
        batch->SetRequestValue(idx, Tuple<Key, Value>{1, key, *value});
      } else {
        Tuple<Key, Value> t;
        t.isvalid = 0;
        return t;
      }
      break;
    case OperationType::REMOVE:
      batch->SetRequestType(idx, REQUEST_REMOVE);
      batch->SetRequestValue(idx, Tuple<Key, Value>{1, key, Value()});
      break;
    case OperationType::INSERT:
      if (value != optional<Value>(nullopt)) {
        batch->SetRequestType(idx, REQUEST_INSERT);
        Tuple<Key, Value> t;
        t.isvalid = 1;
        t.key = key;
        t.value = *value;
        batch->SetRequestValue(idx, t);
      } else {
        Tuple<Key, Value> t;
        t.isvalid = 0;
        return t;
      }
      break;
    }
    // find next q that isnt empty
    // append concurrently to q
    // spin on q signal
    // when signal read value and create boolOrV
    // std::cerr << "About to spin on " << batch->checkSignal()
    //          << " with curr_batch_idx " << curr_batch_idx << " idx " << idx
    //          << " and done " << done[curr_batch_idx] << "\n";
    while (!batch->checkSignal()) {
      if (idx == 0 && !done[curr_batch_idx]) {
        // std::cerr << "Index 0 added the batch\n";
        mtx.lock();
        // std::cerr << "idx is 0 " << batch->CalculateSize() << " "
        //          << inserted[curr_batch_idx] << std::endl;
        if (batch->CalculateSize() ==
            inserted[curr_batch_idx] // &&
                                     //(inserted[curr_batch_idx] == batch_size
                                     //|| (double)(clock() -
                                     //created[curr_batch_idx]) >=
                                     //    TIMEOUT * CLOCKS_PER_SEC)) {
        ) {
          provider.AddBatch(0, batch);
          done[curr_batch_idx] = true;
        }
        mtx.unlock();
      }
      std::this_thread::yield();
    }

    switch (t) {
    case OperationType::GET:
    case OperationType::SET:
    case OperationType::REMOVE:
    case OperationType::INSERT:
      // std::cerr << "[TRACE] got response" << std::endl;
      return batch->GetResponseValues()[idx];
    default:
      Tuple<Key, Value> t;
      t.isvalid = 0;
      return t;
    }
  }

private:
  std::mutex mtx;
  std::vector<Batch<Key, Value> *> curr_batch;
  std::vector<int> inserted;
  std::vector<clock_t> created;
  std::vector<bool> done;
};
} // namespace uhm
#endif