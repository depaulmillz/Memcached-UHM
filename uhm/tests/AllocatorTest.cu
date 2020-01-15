#include "../Allocator.cuh"
#include "TestMacros.cuh"
#include <cuda.h>
#include <iostream>
#include <omp.h>
#include <type_traits>

using namespace std;
using namespace uhm;

int main(int argc, char *argv[]) {
  int rc = 0;

  // ALIGNMENT TEST
  for(int i = 0; i < 10000; i++){
    int* integer = new int();

    gpuErrchk(umallocate(&integer, sizeof(int)));
    if((long)integer % 8 != 0){
      rc = 1;
      cerr << "Non aligned integer ptr with alignment " <<(long)integer % 8 << "\n";
      goto clean;
    }
  }

  


  cerr << "Starting alignment test\n";

  // CONCURRENCY TEST
  cerr << "Starting concurrent test\n";

  int *t1;
  int *t2;
  int *t3;
 
  omp_set_dynamic(0);
  omp_set_num_threads(3);
  int tid;
#pragma omp parallel for private(tid)
  for (tid = 0; tid < 3; tid++) {

    if (tid == 1) {
      gpuErrchk(umallocate(&t1, sizeof(int)));

    } else if (tid == 2) {
      gpuErrchk(umallocate(&t2, sizeof(int)));

    } else {
      gpuErrchk(umallocate(&t3, sizeof(int), 1));
    }
  }

  if (t1 == t2 || t1 == t3 || t2 == t3){
    cerr << "Equal allocations\n";
    rc = 1;
    goto clean;
  }

  clean:
  cleanUpAllocator();

  return rc;
}