#include "../memcached/memcached.h"
#include "Allocator.cuh"
#include "UHM.cuh"
#include "basetypes.cuh"
#include "optional.hh"
#include <cstdlib>
#include <stdint.h>

using namespace uhm;

struct ItemWrapper {
  struct _stritem *i;
  __host__ __device__ ItemWrapper() {}
  __host__ __device__ ItemWrapper(_stritem *rhs) { i = rhs; }
  __host__ __device__ ItemWrapper(const ItemWrapper &rhs) { i = rhs.i; }
  __host__ __device__ ItemWrapper(volatile const ItemWrapper &rhs) {
    i = rhs.i;
  }
  __host__ __device__ bool operator==(const ItemWrapper &rhs) {
    return i == rhs.i;
  }
};

uhm::HashMap<uhm::custr, ItemWrapper> *h = NULL;

char *copyToUM(const char *str) {
  char *newStr;
  int itr = 0;
  unsigned size = 0;
  while (*(str + itr++)) {
    size++;
  }
  size++;
  uhm::umallocate(&newStr, size, 0);
  itr = 0;
  size = 0;
  while (*(str + itr++)) {
    newStr[size] = str[size];
    size++;
  }
  newStr[size] = str[size];
  return newStr;
}
char *copyToUM(char *str, size_t s) {
  char *newStr;

  uhm::umallocate(&newStr, s, 0);
  for (size_t itr = 0; itr < s; itr++) {
    newStr[itr] = str[itr];
  }
  return newStr;
}

extern "C" {
void startMap(size_t size) {
  h = new uhm::HashMap<uhm::custr, ItemWrapper>(size);
}
void endMap() { delete h; }
void resizeMap(unsigned size) { h->resize(size); }
struct _stritem *getFromMap(const char *str) {

  custr c;
  c.str = copyToUM(str);
  uhm::optional<ItemWrapper> resp = h->get(c);
  if (!resp.isNullopt()) {
    ItemWrapper d = *resp;
    return d.i;
  }
  return NULL;
}
struct _stritem *removeFromMap(const char *str) {

  custr c;
  c.str = copyToUM(str);
  uhm::optional<ItemWrapper> resp = h->remove(c);
  if (!resp.isNullopt()) {
    ItemWrapper d = *resp;
    return d.i; // can delete data from map and make copy in normal memory
  }
  return NULL;
}
bool insertIntoMap(const char *str, struct _stritem *d) {
  custr c;
  c.str = copyToUM(str);
  return h->insert(c, ItemWrapper(d));
}
bool setInMap(const char *str, struct _stritem *d) {
  custr c;
  c.str = copyToUM(str);
  return h->set(c, ItemWrapper(d));
}
}