#include "../StdUHM.h"
#include <iostream>
#include <cstring>

int main(int argc, char** argv){
  int rc = 0;
  startMap(10000);
  data_t d;
  d.data = new char[3];
  d.data[0] = 'h';
  d.data[1] = 'i';
  d.data[2] = '\0';
  d.size = 3;

  char* data = new char[3];
  data[0] = 'h';
  data[1] = 'i';
  data[2] = '\0';

  if(!insertIntoMap(data, d)){
    std::cerr << "Error insert didn't work" << std::endl;
    rc = 1;
  }
  char* result;
  std::cerr << (result = getFromMap(data).data) << std::endl;

  if(strcmp(data, result) != 0){
    std::cerr << "Error get didnt work\n";
    rc = 1;
  }

  endMap();
  return rc;
}