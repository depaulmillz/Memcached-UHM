#include "Allocator.cuh"
#include "UHM.cuh"
#include "basetypes.cuh"
#include "optional.hh"
#include <stdint.h>

using namespace uhm;

extern "C" {
struct data_t {
  char *data;
  uint32_t size;
};
}

struct data_t_cpp {
  char* data;
  uint32_t size;
  __host__ __device__ data_t_cpp(): data(NULL), size(0){}
  __host__ __device__ data_t_cpp(const data_t& d){
    this->data = d.data;
    this->size = d.size;
  }
  __host__ __device__ data_t_cpp(const data_t_cpp& d){
    this->data = d.data;
    this->size = d.size;
  }
  __host__ __device__ data_t_cpp(const volatile data_t_cpp& d){
    this->data = d.data;
    this->size = d.size;
  }
  __host__ __device__ bool operator==(const data_t_cpp& d){
    return this->data == d.data && this->size == d.size;
  }
    __host__ __device__ volatile data_t_cpp &operator=(const data_t_cpp &d) volatile {
    this->data = d.data;
    this->size = d.size;
    return *this;
  }


};

uhm::HashMap<uhm::custr, data_t_cpp> *h = NULL;

char *copyToUM(char *str) {
  char *newStr;
  char *itr = str;
  unsigned size = 0;
  while (*itr++) {
    size++;
  }
  size++;
  uhm::umallocate(&newStr, size, 0);
  itr = str;
  size = 0;
  while (*itr++) {
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
void startMap(size_t size) { h = new uhm::HashMap<uhm::custr, data_t_cpp>(size); }
void endMap() { delete h; }
data_t getFromMap(char *str) {

  custr c;
  c.str = copyToUM(str);
  uhm::optional<data_t_cpp> resp = h->get(c);
  if (!resp.isNullopt()) {
    data_t d;
    d.data = (*resp).data;
    d.size = (*resp).size;
    return d;
  }
  data_t d;
  d.data = 0;
  d.size = 0;
  return d;
}
data_t removeFromMap(char *str) {

  custr c;
  c.str = copyToUM(str);
  uhm::optional<data_t_cpp> resp = h->remove(c);
  if (!resp.isNullopt()) {
    data_t d;
    d.data = (*resp).data;
    d.size = (*resp).size;
    return d; // can delete data from map and make copy in normal memory
  }
  data_t d;
  d.data = 0;
  d.size = 0;
  return d;
}
bool insertIntoMap(char *str, data_t d) {
  custr c;
  c.str = copyToUM(str);
  data_t newD;
  newD.data = copyToUM(d.data, d.size);
  return h->insert(c, newD);
}
bool setInMap(char *str, data_t d) {
  custr c;
  c.str = copyToUM(str);
  data_t newD;
  newD.data = copyToUM(d.data, d.size);
  return h->set(c, newD);
}
}