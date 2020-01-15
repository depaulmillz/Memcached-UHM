#ifndef BASETYPES_CUH
#define BASETYPES_CUH
namespace uhm {
struct custr {
  char *str;
  __host__ __device__ custr() { this->str = NULL; }
  __host__ __device__ custr(const custr &other) : str(other.str) {}
  __host__ __device__ custr(const volatile custr &other) : str(other.str) {}
  __host__ __device__ custr(char *str) { this->str = str; }
  __host__ __device__ bool operator==(const custr &other) {
    unsigned idx = 0;
    while (this->str[idx] != '\0' && other.str[idx] != '\0' && this->str[idx] == other.str[idx]) {
      idx++;
    }
    if (this->str[idx] == other.str[idx]) {
      return true;
    }
    return false;
  }
  __host__ __device__ bool operator!=(const custr &other) {
    unsigned idx = 0;
    if (this->str == NULL && other.str != NULL) {
      return true;
    } else if (other.str == NULL) {
      return false;
    }

    while (this->str[idx] != '\0' && other.str[idx] != '\0' && this->str[idx] == other.str[idx]) {
      idx++;
    }
    if (this->str[idx] == other.str[idx]) {
      return false;
    }
    return true;
  }

  __host__ __device__ bool operator<=(const custr &other) {
    unsigned idx = 0;
    if (this->str == NULL && (other.str == NULL || other.str != NULL)) {
      return true;
    } else if (other.str == NULL) {
      return false;
    }
    while (this->str[idx] != '\0' && other.str[idx] != '\0' && this->str[idx] == other.str[idx]) {
      idx++;
    }
    return this->str[idx] - other.str[idx] <= 0;
  }
  __host__ __device__ bool operator>=(const custr &other) {
    unsigned idx = 0;
    if (this->str == NULL && (other.str == NULL || other.str != NULL)) {
      return false;
    } else if (other.str == NULL) {
      return true;
    }

    while (this->str[idx] != '\0' && other.str[idx] != '\0' && this->str[idx] == other.str[idx]) {
      idx++;
    }
    return this->str[idx] - other.str[idx] >= 0;
  }
  /* __host__ __device__ custr &operator=(const custr &rhs) {
     this->str = rhs.str;
     return *this;
   }*/
  __host__ __device__ volatile custr &operator=(const custr &rhs) volatile {
    this->str = rhs.str;
    return *this;
  }
};
} // namespace uhm
#endif