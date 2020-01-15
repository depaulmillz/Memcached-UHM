#ifndef OPTIONAL_CUH
#define OPTIONAL_CUH

#include <iostream>

namespace uhm {

struct nullopt_t {
  explicit constexpr nullopt_t() {}
};

template <typename T> //
class optional {
public:
  optional() = delete;
  constexpr optional(nullopt_t n) : isnull(true) {}
  explicit optional(T data) : data(data), isnull(false) {}
  T operator*() { return data; }
  friend bool operator==(const optional<T> &lhs, const optional<T> &rhs) {
    // std::clog << "Called == " << lhs.isnull << " == " << rhs.isnull <<
    // std::endl;
    if (rhs.isnull == lhs.isnull &&
        (rhs.isnull == true || (T)rhs.data == (T)lhs.data)) {
      return true;
    }
    return false;
  }
  bool isNullopt() { return isnull; }

private:
  T data;
  bool isnull;
};

template <typename T> //
bool operator!=(const optional<T> &lhs, const optional<T> &rhs) {
  return !(lhs == rhs);
}

nullopt_t nullopt;

} // namespace uhm
#endif