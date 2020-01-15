#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
struct data_t {
  char *data;
  uint32_t size;
};
data_t getFromMap(char *str);
void endMap();
void startMap(size_t size);
data_t removeFromMap(char *str);
bool insertIntoMap(char *str, data_t d);
bool setInMap(char *str, data_t d);
#ifdef __cplusplus
}
#endif