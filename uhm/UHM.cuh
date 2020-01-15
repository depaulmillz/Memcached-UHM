/** @file UHM.cuh */
#include "BatchProducer.cuh"
#include "GPUHMBase.cuh"
#include "GPUKVServe.cuh"
#include "optional.hh"
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

namespace uhm {

/**
 * Unified memory hash map served through unified memory
 */
template <typename K, typename V> //
class HashMap {
public:
  /**
   * Create hashmap of given size
   */
  explicit HashMap(size_t size)
      : unified_hash_map(nullptr), provider(1), _size(size), resizing(false) {
    startUHM();
  }
  ~HashMap() { endUHM(); }
  /**
   * Get key from HashMap
   * @param key
   * @returns V
   */
  optional<V> get(K key) {
    while (resizing)
      ;
    auto t = producer.request(OperationType::GET, key, nullopt, provider);
    // std::clog << "tuple t " << t.isvalid << std::endl;
    if (t.isvalid) {
      // std::clog << "Returning a value to get request\n";
      return optional<V>(t.value);
    } else {
      // std::clog << "Returning nullopt to get request\n";
      return nullopt;
    }
  };
  /**
   * Remove key from HashMap
   * @param key
   * @returns V
   */
  optional<V> remove(K key) {
    while (resizing)
      ;

    auto t = producer.request(OperationType::REMOVE, key, nullopt, provider);
    if (t.isvalid) {
      return optional<V>(t.value);
    } else {
      return nullopt;
    }
  };
  /**
   * Insert key,value into HashMap
   * @param key
   * @param value
   * @returns bool
   */
  bool insert(K key, V value) {
    while (resizing)
      ;

    return producer
        .request(OperationType::INSERT, key, optional<V>(value), provider)
        .isvalid;
  };
  /**
   * Set key,value in HashMap
   * @param key
   * @param value
   * @returns bool
   */
  bool set(K key, V value) {
    while (resizing)
      ;

    return producer
        .request(OperationType::SET, key, optional<V>(value), provider)
        .isvalid;
  };

  void resize(unsigned size) {
    while (resizing)
      ;

    if (size > unified_hash_map->GetSize()) {
      signal = true;
      resizing = true;
      __sync_synchronize();
      rwbRunner.join();
      std::vector<Tuple<K, V>> toReinsert;
      toReinsert.reserve(unified_hash_map->GetSize());
      GPUTable<K, V> *t = unified_hash_map->GetTable();
      for (int i = 0; i < t->size; i++) {
        for (int j = 0; j < t->buckets[i].size; j++) {
          toReinsert.push_back(t->buckets[i].elements[j]);
        }
      }
      _size = size;
      startUHM();

      for (auto elm : toReinsert) {
        producer.request(OperationType::INSERT, elm.key, optional<V>(elm.value),
                         provider);
      }

      __sync_synchronize();
      resizing = false;
    }
  }

private:
  /**
   * Starts the UHM serving on the GPU
   */
  void startUHM() {
    unified_hash_map =
        new UnifiedHashTableOnline<K, V>(_size, requests_per_thread);
    signal = false;
    rwbRunner = std::thread([&, this]() {
      while (!this->signal) {
        this->unified_hash_map->readWriteBatch(this->provider);
      }
    });
  }

  /**
   * Ends the UHM serving on the GPU
   */
  void endUHM() {
    std::cerr << "Ending UHM please wait\n";
    signal = true;
    std::cerr << "Joining threads...\n";
    rwbRunner.join();
  }

  UnifiedHashTableOnline<K, V> *unified_hash_map;
  BatchProvider<K, V> provider;
  BatchProducer<K, V> producer;

  size_t _size;
  std::atomic<bool> signal;
  std::atomic<bool> resizing;

  std::thread rwbRunner;
};
} // namespace uhm