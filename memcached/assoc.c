/* -*- Mode: C; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*
 * Hash table
 *
 * The hash function used here is by Bob Jenkins, 1996:
 *    <http://burtleburtle.net/bob/hash/doobs.html>
 *       "By Bob Jenkins, 1996.  bob_jenkins@burtleburtle.net.
 *       You may use this code any way you wish, private, educational,
 *       or commercial.  It's free."
 *
 * The rest of the file is licensed under the BSD license.  See LICENSE.
 */

#include "memcached.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>

static pthread_cond_t maintenance_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t maintenance_lock = PTHREAD_MUTEX_INITIALIZER;

typedef uint32_t ub4;      /* unsigned 4-byte quantities */
typedef unsigned char ub1; /* unsigned 1-byte quantities */

/* how many powers of 2's worth of buckets we use */
unsigned int hashpower = HASHPOWER_DEFAULT;

#define hashsize(n) ((ub4)1 << (n))
#define hashmask(n) (hashsize(n) - 1)

/* Main hash table. This is where we look except during expansion. */
static item **primary_hashtable = 0;

/*
 * Previous hash table. During expansion, we look here for keys that haven't
 * been moved over to the primary yet.
 */
static item **old_hashtable = 0;

/* Flag: Are we in the middle of expanding now? */
static bool expanding = false;

/*
 * During expansion we migrate values with bucket granularity; this is how
 * far we've gotten so far. Ranges from 0 .. hashsize(hashpower - 1) - 1.
 */
static unsigned int expand_bucket = 0;

void assoc_init(const int hashtable_init) {
  if (hashtable_init) {
    hashpower = hashtable_init;
  }
  startMap((unsigned long)hashsize(hashpower));

  STATS_LOCK();
  stats_state.hash_power_level = hashpower;
  stats_state.hash_bytes = hashsize(hashpower) * sizeof(void *);
  STATS_UNLOCK();
}

item *assoc_find(const char *key, const size_t nkey, const uint32_t hv) {

  item *ret = getFromMap(key);

  MEMCACHED_ASSOC_FIND(key, nkey, 0);
  return ret;
}

/* grows the hashtable to the next power of 2. */
static void assoc_expand(void) {
  hashpower++;
  fprintf(stderr, "HASH SIZE %d", hashsize(hashpower));
  resizeMap(hashsize(hashpower));
}

void assoc_start_expand(uint64_t curr_items) {
  if (pthread_mutex_trylock(&maintenance_lock) == 0) {
    // fprintf(stderr, "ASSOC EXPAND CALLED\n");
    if (curr_items > (hashsize(hashpower) * 3) / 2 &&
        hashpower < HASHPOWER_MAX) {
      pthread_cond_signal(&maintenance_cond);
    }
    pthread_mutex_unlock(&maintenance_lock);
  }
}

/* Note: this isn't an assoc_update.  The key must not already exist to call
 * this */
int assoc_insert(item *it, const uint32_t hv) {
  MEMCACHED_ASSOC_INSERT(ITEM_key(it), it->nkey);
  return (int)insertIntoMap(ITEM_key(it), it);
}

void assoc_delete(const char *key, const size_t nkey, const uint32_t hv) {
  removeFromMap(key);
  MEMCACHED_ASSOC_DELETE(key, nkey);
}

static volatile int do_run_maintenance_thread = 1;

#define DEFAULT_HASH_BULK_MOVE 1
int hash_bulk_move = DEFAULT_HASH_BULK_MOVE;

static void *assoc_maintenance_thread(void *arg) {

  mutex_lock(&maintenance_lock);
  while (do_run_maintenance_thread) {
    int ii = 0;

    /* There is only one expansion thread, so no need to global lock. */
    for (ii = 0; ii < hash_bulk_move && expanding; ++ii) {
      item *it, *next;
      unsigned int bucket;
      void *item_lock = NULL;

      /* bucket = hv & hashmask(hashpower) =>the bucket of hash table
       * is the lowest N bits of the hv, and the bucket of item_locks is
       *  also the lowest M bits of hv, and N is greater than M.
       *  So we can process expanding with only one item_lock. cool! */
      if ((item_lock = item_trylock(expand_bucket))) {
        for (it = old_hashtable[expand_bucket]; NULL != it; it = next) {
          next = it->h_next;
          bucket = hash(ITEM_key(it), it->nkey) & hashmask(hashpower);
          it->h_next = primary_hashtable[bucket];
          primary_hashtable[bucket] = it;
        }

        old_hashtable[expand_bucket] = NULL;

        expand_bucket++;
        if (expand_bucket == hashsize(hashpower - 1)) {
          expanding = false;
          free(old_hashtable);
          STATS_LOCK();
          stats_state.hash_bytes -= hashsize(hashpower - 1) * sizeof(void *);
          stats_state.hash_is_expanding = false;
          STATS_UNLOCK();
          if (settings.verbose > 1)
            fprintf(stderr, "Hash table expansion done\n");
        }

      } else {
        usleep(10 * 1000);
      }

      if (item_lock) {
        item_trylock_unlock(item_lock);
        item_lock = NULL;
      }
    }

    if (!expanding) {
      /* We are done expanding.. just wait for next invocation */
      pthread_cond_wait(&maintenance_cond, &maintenance_lock);
      /* assoc_expand() swaps out the hash table entirely, so we need
       * all threads to not hold any references related to the hash
       * table while this happens.
       * This is instead of a more complex, possibly slower algorithm to
       * allow dynamic hash table expansion without causing significant
       * wait times.
       */
      if (do_run_maintenance_thread) {
        pause_threads(PAUSE_ALL_THREADS);
        assoc_expand();
        pause_threads(RESUME_ALL_THREADS);
      }
    }
  }
  mutex_unlock(&maintenance_lock);
  return NULL;
}

static pthread_t maintenance_tid;

int start_assoc_maintenance_thread() {
  int ret;
  char *env = getenv("MEMCACHED_HASH_BULK_MOVE");
  if (env != NULL) {
    hash_bulk_move = atoi(env);
    if (hash_bulk_move == 0) {
      hash_bulk_move = DEFAULT_HASH_BULK_MOVE;
    }
  }

  if ((ret = pthread_create(&maintenance_tid, NULL, assoc_maintenance_thread,
                            NULL)) != 0) {
    fprintf(stderr, "Can't create thread: %s\n", strerror(ret));
    return -1;
  }
  return 0;
}

void stop_assoc_maintenance_thread() {
  mutex_lock(&maintenance_lock);
  do_run_maintenance_thread = 0;
  pthread_cond_signal(&maintenance_cond);
  mutex_unlock(&maintenance_lock);

  /* Wait for the maintenance thread to stop */
  pthread_join(maintenance_tid, NULL);
}